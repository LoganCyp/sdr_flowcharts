#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-Pole Receiver — QPSK (ch0) + BPSK (ch1)
All QPSK parameters match the proven working GRC at 3 GHz exactly.

MIMO XPIC: Pure CMA with two targeted improvements:
  1. Separate step sizes — direct taps (w00, w11) use mu_direct,
     cross taps (w01, w10) use a smaller mu_cross. The cross taps
     are doing interference cancellation and are more sensitive to
     noise injection from the interfering channel.
  2. Tap leakage — all taps are gently pulled toward the identity
     matrix every sample. This prevents drift when one channel
     carries no signal (e.g. BPSK Tx off) and keeps cross taps
     from fitting to noise.

RFML: Binary IQ classifier (1D CNN) trained on real captured data.
  - Input: 1024 complex64 samples → 2-channel (I,Q) tensor
  - Output: BPSK or QPSK with confidence
  - No scaler needed — raw IQ in, prediction out

Signal flow:
  USRP ch0 -> AGC -> FLL -> PFB(osps=2) -> MIMO XPIC -> ModClassifier -> Costas(4) -> QPSK decode
  USRP ch1 -> AGC -> FLL -> PFB(osps=2) -> MIMO XPIC -> ModClassifier -> Costas(2) -> BPSK decode
"""

import os
import sys
import signal
import io
import json
import tempfile
import numpy as np
import numba as nb
import pmt
from PyQt5 import Qt
from PyQt5.QtCore import pyqtSignal, QObject
import torch
import torch.nn as tnn
from gnuradio import analog, blocks, digital, gr, uhd, pdu
from gnuradio.filter import firdes

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARN] PIL/Pillow not installed — image validation will be weak")

# =============================================================================
# RFML Config — point this at E:\04_06_26_Model (or wherever you saved it)
# =============================================================================
RFML_MODEL_DIR   = r"E:\04_06_26_Model"
RFML_SLICE_LEN   = 1024       # must match training
RFML_INTERVAL    = 50000      # classify every N samples

JPEG_START = b"\xFF\xD8"
JPEG_END   = b"\xFF\xD9"
MAX_BUF_SIZE = 2_000_000


class SignalProxy(QObject):
    image_received = pyqtSignal(str)


##################################################
# Packet Counter — counts CRC-valid packets
##################################################
class PacketCounter(gr.basic_block):
    def __init__(self, channel=''):
        gr.basic_block.__init__(self, name=f"PktCounter {channel}",
                                in_sig=None, out_sig=None)
        self.message_port_register_in(pmt.intern("in"))
        self.message_port_register_out(pmt.intern("out"))
        self.set_msg_handler(pmt.intern("in"), self._handle)
        self.channel = channel
        self.count = 0
        self.total_bytes = 0

    def _handle(self, msg):
        self.count += 1
        try:
            vec = pmt.cdr(msg)
            self.total_bytes += len(pmt.u8vector_elements(vec))
        except Exception:
            pass

        if self.count % 50 == 1:
            print(f"[{self.channel}] CRC-valid packet #{self.count} "
                  f"({self.total_bytes} total bytes)")

        self.message_port_pub(pmt.intern("out"), msg)


##################################################
# Pre-CRC Packet Counter
##################################################
class PreCrcCounter(gr.sync_block):
    def __init__(self, channel=''):
        gr.sync_block.__init__(self,
                               name=f"PreCRC {channel}",
                               in_sig=[np.uint8],
                               out_sig=[np.uint8])
        self.channel = channel
        self.count = 0
        self._last_offset = -1

    def work(self, input_items, output_items):
        n = len(input_items[0])
        output_items[0][:n] = input_items[0][:n]

        tags = self.get_tags_in_window(0, 0, n)
        for tag in tags:
            if pmt.symbol_to_string(tag.key) == "packet_len":
                offset = tag.offset
                if offset != self._last_offset:
                    self.count += 1
                    self._last_offset = offset

        return n


##################################################
# Robust Image Recovery
##################################################
class ImageRecoveryBlock(gr.basic_block):
    def __init__(self, out_jpg='recovered.jpg', channel=''):
        gr.basic_block.__init__(self, name=f"ImageRecovery {channel}",
                                in_sig=None, out_sig=None)
        self.message_port_register_in(pmt.intern("pdus"))
        self.set_msg_handler(pmt.intern("pdus"), self._handle)

        self.out_jpg = out_jpg
        self.channel = channel
        self.buf = bytearray()
        self.proxy = SignalProxy()
        self.pkt_count = 0
        self.valid_count = 0
        self.reject_count = 0

    def _handle(self, msg):
        try:
            vec = pmt.cdr(msg)
            data = bytes(pmt.u8vector_elements(vec))
        except Exception:
            return

        self.pkt_count += 1
        self.buf.extend(data)

        if len(self.buf) > MAX_BUF_SIZE:
            self.buf = self.buf[-MAX_BUF_SIZE:]

        self._search()

    def _search(self):
        pos = 0
        while True:
            s = self.buf.find(JPEG_START, pos)
            if s < 0:
                break
            e = self.buf.find(JPEG_END, s + 2)
            if e < 0:
                if s > 0:
                    del self.buf[:s]
                return
            candidate = bytes(self.buf[s:e + 2])
            pos = e + 2

            if len(candidate) < 2000:
                continue

            if self._validate(candidate):
                self.valid_count += 1
                self._atomic_save(candidate)
                print(f"[{self.channel}] VALID image #{self.valid_count}: "
                      f"{len(candidate)} bytes")
                self.proxy.image_received.emit(self.out_jpg)
            else:
                self.reject_count += 1

        if pos > 0:
            del self.buf[:pos]

    def _validate(self, b):
        if not HAS_PIL:
            return len(b) > 5000
        try:
            im = Image.open(io.BytesIO(b))
            im.load()
            w, h = im.size
            return w >= 16 and h >= 16
        except Exception:
            return False

    def _atomic_save(self, data):
        out_dir = os.path.dirname(os.path.abspath(self.out_jpg))
        try:
            fd, tmp = tempfile.mkstemp(suffix='.jpg', dir=out_dir)
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
            os.replace(tmp, self.out_jpg)
        except Exception as ex:
            print(f"[{self.channel}] Save error: {ex}")
            try:
                os.unlink(tmp)
            except Exception:
                pass


##################################################
# NUMBA-Compiled CMA XPIC
# with tap leakage + separate direct/cross step sizes
##################################################

@nb.jit(nopython=True, fastmath=True, nogil=True, cache=True)
def numba_cma_xpic(in0, in1, out0, out1,
                    mu_direct, mu_cross, rho, leak,
                    w00, w01, w10, w11,
                    err_acc0, err_acc1, err_count):
    n = in0.shape[0]

    for i in range(n):
        x0 = in0[i]
        x1 = in1[i]

        y0 = w00 * x0 + w01 * x1
        y1 = w10 * x0 + w11 * x1

        out0[i] = y0
        out1[i] = y1

        mag_sq0 = y0.real * y0.real + y0.imag * y0.imag
        mag_sq1 = y1.real * y1.real + y1.imag * y1.imag

        # Soft divergence protection
        if mag_sq0 > 4.0 or mag_sq1 > 4.0:
            w00 = 0.9 * w00 + 0.1 * (1.0 + 0j)
            w01 = 0.9 * w01
            w10 = 0.9 * w10
            w11 = 0.9 * w11 + 0.1 * (1.0 + 0j)
            continue

        err0 = y0 * (mag_sq0 - 1.0)
        err1 = y1 * (mag_sq1 - 1.0)

        cross = y0 * y1.conjugate()
        err0 = err0 + rho * y1 * cross.conjugate()
        err1 = err1 + rho * y0 * cross

        err_acc0 += abs(err0)
        err_acc1 += abs(err1)
        err_count += 1

        x0c = x0.conjugate()
        x1c = x1.conjugate()

        w00 -= mu_direct * err0 * x0c
        w01 -= mu_cross  * err0 * x1c
        w10 -= mu_cross  * err1 * x0c
        w11 -= mu_direct * err1 * x1c

        w00 = w00 * (1.0 - leak) + leak * (1.0 + 0j)
        w01 = w01 * (1.0 - leak)
        w10 = w10 * (1.0 - leak)
        w11 = w11 * (1.0 - leak) + leak * (1.0 + 0j)

    return w00, w01, w10, w11, err_acc0, err_acc1, err_count


class mimo_xpic_2x2(gr.sync_block):
    def __init__(self, mu_direct=1e-4, mu_cross=3e-5,
                 rho=0.05, leak=1e-6):
        gr.sync_block.__init__(self,
                               name="MIMO XPIC 2x2 (CMA+leak)",
                               in_sig=[np.complex64, np.complex64],
                               out_sig=[np.complex64, np.complex64])
        self.mu_direct = mu_direct
        self.mu_cross = mu_cross
        self.rho = rho
        self.leak = leak

        self.w00 = 1.0 + 0j
        self.w01 = 0.0 + 0j
        self.w10 = 0.0 + 0j
        self.w11 = 1.0 + 0j

        self.err_acc0 = 0.0
        self.err_acc1 = 0.0
        self.err_count = 0

    def work(self, input_items, output_items):
        (self.w00, self.w01, self.w10, self.w11,
         self.err_acc0, self.err_acc1, self.err_count) = numba_cma_xpic(
            input_items[0], input_items[1],
            output_items[0], output_items[1],
            self.mu_direct, self.mu_cross, self.rho, self.leak,
            self.w00, self.w01, self.w10, self.w11,
            self.err_acc0, self.err_acc1, self.err_count
        )
        return len(input_items[0])

    def get_avg_error(self):
        if self.err_count == 0:
            return 0.0, 0.0
        avg0 = self.err_acc0 / self.err_count
        avg1 = self.err_acc1 / self.err_count
        self.err_acc0 = 0.0
        self.err_acc1 = 0.0
        self.err_count = 0
        return avg0, avg1


##################################################
# IQClassifier — binary BPSK/QPSK CNN
# Must match the architecture from training
##################################################

class IQClassifier(tnn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = tnn.Sequential(
            tnn.Conv1d(2, 32, kernel_size=7, padding=3),
            tnn.BatchNorm1d(32),
            tnn.ReLU(),
            tnn.MaxPool1d(4),

            tnn.Conv1d(32, 64, kernel_size=5, padding=2),
            tnn.BatchNorm1d(64),
            tnn.ReLU(),
            tnn.MaxPool1d(4),

            tnn.Conv1d(64, 128, kernel_size=3, padding=1),
            tnn.BatchNorm1d(128),
            tnn.ReLU(),
            tnn.AdaptiveAvgPool1d(1),
        )
        self.classifier = tnn.Sequential(
            tnn.Flatten(),
            tnn.Linear(128, 32),
            tnn.ReLU(),
            tnn.Dropout(0.3),
            tnn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


##################################################
# Modulation Classifier Block — GNU Radio pass-through
#
# Accumulates IQ samples, grabs RFML_SLICE_LEN (1024)
# every classify_interval samples, runs the CNN,
# stores prediction + confidence for the GUI to read.
##################################################

class ModulationClassifierBlock(gr.sync_block):
    def __init__(self, model, labels, device, channel='',
                 classify_interval=50000, slice_len=1024):
        gr.sync_block.__init__(self,
                               name=f"ModClassifier {channel}",
                               in_sig=[np.complex64],
                               out_sig=[np.complex64])
        self.ml_model = model
        self.labels = labels        # {0: "BPSK", 1: "QPSK"}
        self.device = device
        self.channel = channel
        self.prediction = "---"
        self.confidence = 0.0
        self.sample_count = 0
        self.classify_interval = classify_interval
        self.slice_len = slice_len

        # Ring buffer to accumulate enough samples
        self.iq_buf = np.zeros(slice_len, dtype=np.complex64)
        self.buf_idx = 0

    def work(self, input_items, output_items):
        n = len(input_items[0])
        output_items[0][:n] = input_items[0][:n]

        # Fill ring buffer
        inp = input_items[0][:n]
        space = self.slice_len - self.buf_idx
        if n >= space:
            self.iq_buf[self.buf_idx:] = inp[:space]
            self.buf_idx = 0
        else:
            self.iq_buf[self.buf_idx:self.buf_idx + n] = inp
            self.buf_idx += n

        self.sample_count += n
        if self.sample_count >= self.classify_interval:
            self.sample_count = 0
            self._classify(self.iq_buf.copy())

        return n

    def _classify(self, iq_complex):
        try:
            # Shape: (2, 1024) — I and Q as separate channels
            x = np.stack([iq_complex.real, iq_complex.imag],
                         axis=0).astype(np.float32)
            x = torch.from_numpy(x).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.ml_model(x)
                probs = torch.softmax(logits, dim=1)
                idx = probs.argmax(1).item()
                conf = probs[0, idx].item()

            self.prediction = self.labels[idx]
            self.confidence = conf
        except Exception as e:
            print(f"[{self.channel}] Classify error: {e}")


##################################################
# Load RFML model from saved directory
##################################################

def load_rfml(model_dir):
    """
    Load the trained binary classifier from model_dir.
    Expects:
        model_dir/iq_classifier_best.pth
        model_dir/config.json
    Returns: (model, labels_dict, device)
    """
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "iq_classifier_best.pth")

    with open(config_path, "r") as f:
        config = json.load(f)

    # labels: {0: "BPSK", 1: "QPSK"}
    labels = {int(k): v for k, v in config["labels"].items()}
    num_classes = config["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IQClassifier(num_classes=num_classes)
    model.load_state_dict(
        torch.load(weights_path, map_location=device)
    )
    model.to(device)
    model.eval()

    print(f"[RFML] IQClassifier loaded — {num_classes} classes "
          f"({', '.join(labels.values())}), device={device}")

    return model, labels, device


##################################################
# Main flowgraph
##################################################
class dual_pole_rx(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "Dual-Pole Rx",
                              catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Dual-Pole Receiver — 3.0 GHz")
        self.resize(950, 620)

        main_layout = Qt.QVBoxLayout(self)

        ##################################################
        # Variables
        ##################################################
        sps         = 4
        samp_rate   = 1e6
        freq        = 2.45e9
        excess_bw   = 0.35

        rx_gain_ch0 = 20
        rx_gain_ch1 = 20

        qpsk_access = '11100001010110101110100010010011'
        bpsk_access = '10010110110110100101000111011001'

        rrc_taps = firdes.root_raised_cosine(32, 32, 1.0 / 1.0, 0.35, 11 * 32)

        ##################################################
        # GUI
        ##################################################

        img_row = Qt.QHBoxLayout()

        # --- CH0: QPSK ---
        q_grp = Qt.QGroupBox("CH0 — QPSK")
        q_lay = Qt.QVBoxLayout()
        q_grp.setLayout(q_lay)
        self.q_img = Qt.QLabel("Awaiting QPSK...")
        self.q_img.setAlignment(Qt.Qt.AlignCenter)
        self.q_img.setStyleSheet(
            "font-size: 18px; color: #555; border: 2px dashed #aaa; "
            "background: #eee;")
        self.q_img.setMinimumSize(400, 400)
        q_lay.addWidget(self.q_img)

        self.q_gain_lbl = Qt.QLabel(f"Gain: {rx_gain_ch0} dB")
        q_lay.addWidget(self.q_gain_lbl)
        self.q_gain = Qt.QSlider(Qt.Qt.Horizontal)
        self.q_gain.setRange(0, 76)
        self.q_gain.setValue(rx_gain_ch0)
        self.q_gain.valueChanged.connect(self._set_q_gain)
        q_lay.addWidget(self.q_gain)
        img_row.addWidget(q_grp)

        # --- CH1: BPSK ---
        b_grp = Qt.QGroupBox("CH1 — BPSK")
        b_lay = Qt.QVBoxLayout()
        b_grp.setLayout(b_lay)
        self.b_img = Qt.QLabel("Awaiting BPSK...")
        self.b_img.setAlignment(Qt.Qt.AlignCenter)
        self.b_img.setStyleSheet(
            "font-size: 18px; color: #555; border: 2px dashed #aaa; "
            "background: #eee;")
        self.b_img.setMinimumSize(400, 400)
        b_lay.addWidget(self.b_img)

        self.b_gain_lbl = Qt.QLabel(f"Gain: {rx_gain_ch1} dB")
        b_lay.addWidget(self.b_gain_lbl)
        self.b_gain = Qt.QSlider(Qt.Qt.Horizontal)
        self.b_gain.setRange(0, 76)
        self.b_gain.setValue(rx_gain_ch1)
        self.b_gain.valueChanged.connect(self._set_b_gain)
        b_lay.addWidget(self.b_gain)
        img_row.addWidget(b_grp)

        main_layout.addLayout(img_row)

        # --- XPIC status row ---
        xpic_row = Qt.QHBoxLayout()
        self.xpic_status = Qt.QLabel("XPIC: initializing...")
        self.xpic_status.setStyleSheet(
            "font-size: 11px; color: #4a7; padding: 4px; "
            "font-family: monospace;")
        xpic_row.addWidget(self.xpic_status)
        main_layout.addLayout(xpic_row)

        # --- Modulation prediction row ---
        mod_row = Qt.QHBoxLayout()
        self.mod_status = Qt.QLabel("Modulation: waiting...")
        self.mod_status.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #27e; padding: 4px; "
            "font-family: monospace;")
        mod_row.addWidget(self.mod_status)
        main_layout.addLayout(mod_row)

        self.status = Qt.QLabel("Initializing...")
        self.status.setStyleSheet("font-size: 11px; color: #888; padding: 4px;")
        main_layout.addWidget(self.status)

        ##################################################
        # USRP — 2-channel B210
        ##################################################

        self.usrp = uhd.usrp_source(
            ",".join(("", '')),
            uhd.stream_args(cpu_format="fc32", args='',
                            channels=list(range(0, 2))),
        )
        self.usrp.set_samp_rate(samp_rate)

        self.usrp.set_center_freq(freq, 0)
        self.usrp.set_antenna("RX2", 0)
        self.usrp.set_gain(rx_gain_ch0, 0)

        self.usrp.set_center_freq(freq, 1)
        self.usrp.set_antenna("RX2", 1)
        self.usrp.set_gain(rx_gain_ch1, 1)

        print(f"[USRP] 2-ch mode, {samp_rate/1e6} MS/s, "
              f"freq={freq/1e9} GHz, "
              f"gain=[{rx_gain_ch0}, {rx_gain_ch1}]")

        ##################################################
        # AGC — exact from working GRC
        ##################################################

        self.agc_0 = analog.agc_cc(1e-4, 1.0, 1.0, 4000)
        self.agc_1 = analog.agc_cc(1e-4, 1.0, 1.0, 4000)

        ##################################################
        # FLL — exact from working GRC
        ##################################################

        self.fll_0 = digital.fll_band_edge_cc(sps, excess_bw, 44, 0.0628)
        self.fll_1 = digital.fll_band_edge_cc(sps, excess_bw, 44, 0.0628)

        ##################################################
        # PFB Clock Sync — exact from working GRC
        ##################################################

        self.pfb_0 = digital.pfb_clock_sync_ccf(
            sps, 0.0628, rrc_taps, 32, 16, 1.5, 1)
        self.pfb_1 = digital.pfb_clock_sync_ccf(
            sps, 0.0628, rrc_taps, 32, 16, 1.5, 1)

        ##################################################
        # MIMO XPIC — CMA with leakage + split step sizes
        ##################################################

        self.mimo = mimo_xpic_2x2(
            mu_direct=1e-4,
            mu_cross=3e-5,
            rho=0.05,
            leak=1e-6
        )

        ##################################################
        # RFML — Binary IQ Classifier (BPSK / QPSK)
        ##################################################

        ml_model, ml_labels, ml_device = load_rfml(RFML_MODEL_DIR)

        self.mod_clf_0 = ModulationClassifierBlock(
            ml_model, ml_labels, ml_device,
            channel='CH0', classify_interval=RFML_INTERVAL,
            slice_len=RFML_SLICE_LEN)
        self.mod_clf_1 = ModulationClassifierBlock(
            ml_model, ml_labels, ml_device,
            channel='CH1', classify_interval=RFML_INTERVAL,
            slice_len=RFML_SLICE_LEN)

        ##################################################
        # Costas — exact from working GRC
        ##################################################

        self.costas_0 = digital.costas_loop_cc(0.0628, 4, False)
        self.costas_1 = digital.costas_loop_cc(0.0628, 2, False)

        ##################################################
        # QPSK decode — exact from working GRC
        ##################################################

        self.dec_q   = digital.constellation_decoder_cb(
                           digital.constellation_qpsk().base())
        self.diff_q  = digital.diff_decoder_bb(4, digital.DIFF_DIFFERENTIAL)
        self.unp_q   = blocks.unpack_k_bits_bb(2)
        self.corr_q  = digital.correlate_access_code_bb_ts(
                           qpsk_access, 2, "packet_len")
        self.rep_q   = blocks.repack_bits_bb(
                           1, 8, "packet_len", True, gr.GR_MSB_FIRST)
        self.crc_q   = digital.crc32_bb(True, "packet_len", True)
        self.pdu_q   = pdu.tagged_stream_to_pdu(
                           gr.types.byte_t, 'packet_len')

        self.cnt_q   = PacketCounter(channel='QPSK')
        self.precrc_q = PreCrcCounter(channel='QPSK')
        self.rec_q   = ImageRecoveryBlock(
                           out_jpg='qpsk_recovered.jpg', channel='QPSK')

        ##################################################
        # BPSK decode
        ##################################################

        self.dec_b   = digital.constellation_decoder_cb(
                           digital.constellation_bpsk().base())
        self.diff_b  = digital.diff_decoder_bb(2, digital.DIFF_DIFFERENTIAL)
        self.corr_b  = digital.correlate_access_code_bb_ts(
                           bpsk_access, 2, "packet_len")
        self.rep_b   = blocks.repack_bits_bb(
                           1, 8, "packet_len", True, gr.GR_MSB_FIRST)
        self.crc_b   = digital.crc32_bb(True, "packet_len", True)
        self.pdu_b   = pdu.tagged_stream_to_pdu(
                           gr.types.byte_t, 'packet_len')

        self.cnt_b   = PacketCounter(channel='BPSK')
        self.precrc_b = PreCrcCounter(channel='BPSK')
        self.rec_b   = ImageRecoveryBlock(
                           out_jpg='bpsk_recovered.jpg', channel='BPSK')

        ##################################################
        # Null sink on ch1 — scheduler insurance
        ##################################################

        self.null_1 = blocks.null_sink(gr.sizeof_gr_complex)

        ##################################################
        # Qt signals + timer
        ##################################################

        self.rec_q.proxy.image_received.connect(self._show_q)
        self.rec_b.proxy.image_received.connect(self._show_b)

        self.timer = Qt.QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(2000)

        ##################################################
        # Connections
        ##################################################

        # === CH0: QPSK ===
        self.connect((self.usrp, 0), (self.agc_0, 0))
        self.connect((self.agc_0, 0), (self.fll_0, 0))
        self.connect((self.fll_0, 0), (self.pfb_0, 0))

        self.connect((self.pfb_0, 0), (self.mimo, 0))
        self.connect((self.mimo, 0),  (self.mod_clf_0, 0))
        self.connect((self.mod_clf_0, 0), (self.costas_0, 0))

        self.connect((self.costas_0, 0), (self.dec_q, 0))
        self.connect((self.dec_q, 0),    (self.diff_q, 0))
        self.connect((self.diff_q, 0),   (self.unp_q, 0))
        self.connect((self.unp_q, 0),    (self.corr_q, 0))
        self.connect((self.corr_q, 0),   (self.rep_q, 0))
        self.connect((self.rep_q, 0),    (self.precrc_q, 0))
        self.connect((self.precrc_q, 0), (self.crc_q, 0))
        self.connect((self.crc_q, 0),    (self.pdu_q, 0))
        self.msg_connect((self.pdu_q, 'pdus'), (self.cnt_q, 'in'))
        self.msg_connect((self.cnt_q, 'out'),  (self.rec_q, 'pdus'))

        # === CH1: BPSK ===
        self.connect((self.usrp, 1), (self.agc_1, 0))
        self.connect((self.agc_1, 0), (self.fll_1, 0))
        self.connect((self.fll_1, 0), (self.pfb_1, 0))

        self.connect((self.pfb_1, 0), (self.mimo, 1))
        self.connect((self.mimo, 1),  (self.mod_clf_1, 0))
        self.connect((self.mod_clf_1, 0), (self.costas_1, 0))

        self.connect((self.costas_1, 0), (self.null_1, 0))

        self.connect((self.costas_1, 0), (self.dec_b, 0))
        self.connect((self.dec_b, 0),    (self.diff_b, 0))
        self.connect((self.diff_b, 0),   (self.corr_b, 0))
        self.connect((self.corr_b, 0),   (self.rep_b, 0))
        self.connect((self.rep_b, 0),    (self.precrc_b, 0))
        self.connect((self.precrc_b, 0), (self.crc_b, 0))
        self.connect((self.crc_b, 0),    (self.pdu_b, 0))
        self.msg_connect((self.pdu_b, 'pdus'), (self.cnt_b, 'in'))
        self.msg_connect((self.cnt_b, 'out'),  (self.rec_b, 'pdus'))

        print("[INIT] Flowgraph built (XPIC: CMA + leakage + split mu). Starting...")

    ##################################################
    # Callbacks
    ##################################################

    def _set_q_gain(self, v):
        self.usrp.set_gain(v, 0)
        self.q_gain_lbl.setText(f"Gain: {v} dB")

    def _set_b_gain(self, v):
        self.usrp.set_gain(v, 1)
        self.b_gain_lbl.setText(f"Gain: {v} dB")

    def _show_q(self, path):
        px = Qt.QPixmap(path)
        if not px.isNull():
            self.q_img.setPixmap(
                px.scaled(self.q_img.size(),
                          Qt.Qt.KeepAspectRatio,
                          Qt.Qt.SmoothTransformation))
            self.q_img.setStyleSheet("border: none; background: transparent;")
        else:
            print(f"[QPSK] QPixmap FAILED on {path}")

    def _show_b(self, path):
        px = Qt.QPixmap(path)
        if not px.isNull():
            self.b_img.setPixmap(
                px.scaled(self.b_img.size(),
                          Qt.Qt.KeepAspectRatio,
                          Qt.Qt.SmoothTransformation))
            self.b_img.setStyleSheet("border: none; background: transparent;")
        else:
            print(f"[BPSK] QPixmap FAILED on {path}")

    def _tick(self):
        q_total = self.precrc_q.count
        q_good  = self.cnt_q.count
        b_total = self.precrc_b.count
        b_good  = self.cnt_b.count

        if q_total > 0:
            q_per = 100.0 * (1.0 - q_good / q_total)
            q_per_str = f"{q_per:.1f}% ({q_good}/{q_total})"
        else:
            q_per_str = "-"

        if b_total > 0:
            b_per = 100.0 * (1.0 - b_good / b_total)
            b_per_str = f"{b_per:.1f}% ({b_good}/{b_total})"
        else:
            b_per_str = "-"

        self.status.setText(
            f"PER  QPSK: {q_per_str}  |  BPSK: {b_per_str}")

        # Modulation predictions from binary classifier
        pred0 = self.mod_clf_0.prediction
        conf0 = self.mod_clf_0.confidence
        pred1 = self.mod_clf_1.prediction
        conf1 = self.mod_clf_1.confidence
        self.mod_status.setText(
            f"RFML  CH0: {pred0} ({conf0:.0%})  |  "
            f"CH1: {pred1} ({conf1:.0%})")

        avg0, avg1 = self.mimo.get_avg_error()
        w00 = self.mimo.w00
        w01 = self.mimo.w01
        w10 = self.mimo.w10
        w11 = self.mimo.w11
        iso0 = -20 * np.log10(max(abs(w01) / max(abs(w00), 1e-12), 1e-12))
        iso1 = -20 * np.log10(max(abs(w10) / max(abs(w11), 1e-12), 1e-12))
        self.xpic_status.setText(
            f"XPI: {iso0:.1f}/{iso1:.1f} dB  |  "
            f"err: {avg0:.4f}/{avg1:.4f}")

    def closeEvent(self, event):
        self.timer.stop()
        self.stop()
        self.wait()
        event.accept()


def main():
    qapp = Qt.QApplication(sys.argv)
    tb = dual_pole_rx()
    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.timer.stop()
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("[RUN] Flowgraph running (XPIC: pure CMA + leakage + split mu).")
    print("[RUN] RFML: Binary IQ classifier (BPSK/QPSK) active.")
    print("[RUN] XPIC status bar shows tap magnitudes and cross-pol isolation.")

    qapp.exec_()


if __name__ == '__main__':
    main()
