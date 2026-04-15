#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-Pole Receiver — QPSK (ch0) + BPSK (ch1)
Integrated with Real-Time RadioLSTM Modulation Classification.

Signal flow:
  USRP -> AGC -> FLL -> PFB -> MIMO XPIC -> RadioLSTM Classifier -> Costas -> Decode
"""

import os
import sys
import signal
import io
import tempfile
import numpy as np
import numba as nb
import pmt
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from PyQt5 import Qt
from PyQt5.QtCore import pyqtSignal, QObject
from gnuradio import analog, blocks, digital, gr, uhd, pdu
from gnuradio.filter import firdes

# --- RFML Model Definition (Must match tester.py) ---
class RadioLSTM(nn.Module):
    def __init__(self, input_dim=2, num_classes=11):
        super(RadioLSTM, self).__init__()
        self.lstm1      = nn.LSTM(input_dim, 128, batch_first=True, bidirectional=True)
        self.dropout1   = nn.Dropout(0.3)
        self.lstm2      = nn.LSTM(256, 128, batch_first=True)
        self.batchnorm  = nn.BatchNorm1d(256)
        self.fc         = nn.Linear(256, 128)
        self.dropout2   = nn.Dropout(0.2)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x     = self.dropout1(x)
        x, _ = self.lstm2(x)
        x_mean = x.mean(dim=1)
        x_max  = x.max(dim=1).values
        x      = torch.cat([x_mean, x_max], dim=1)
        x      = self.batchnorm(x)
        return self.classifier(self.dropout2(F.relu(self.fc(x))))

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

JPEG_START = b"\xFF\xD8"
JPEG_END   = b"\xFF\xD9"
MAX_BUF_SIZE = 2_000_000

class SignalProxy(QObject):
    image_received = pyqtSignal(str)

##################################################
# Real-Time Modulation Classifier Block
##################################################
class ModulationClassifierBlock(gr.sync_block):
    def __init__(self, model_dir="artifacts", device='cpu'):
        gr.sync_block.__init__(self, name="ModClassifier", in_sig=[np.complex64], out_sig=None)
        
        # Load RFML Artifacts
        self.classes = joblib.load(os.path.join(model_dir, "class_labels.joblib"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        
        self.device = torch.device(device)
        self.model = RadioLSTM(num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.seq_len = 128
        self.buffer = np.zeros(self.seq_len, dtype=np.complex64)
        self.current_idx = 0
        
        self.latest_pred = "Initializing..."
        self.confidence = 0.0

    def work(self, input_items, output_items):
        in0 = input_items[0]
        for val in in0:
            self.buffer[self.current_idx] = val
            self.current_idx += 1
            if self.current_idx >= self.seq_len:
                self._classify()
                self.current_idx = 0
        return len(in0)

    def _classify(self):
        # Preprocess matching tester.py logic
        iq = np.stack([self.buffer.real, self.buffer.imag], axis=1).astype(np.float32)
        iq_scaled = self.scaler.transform(iq)
        x = torch.tensor(iq_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            idx = probs.argmax().item()
            self.latest_pred = self.classes[idx]
            self.confidence = probs[0, idx].item()

##################################################
# Original Utility Blocks (Packet Counters & Image Recovery)
##################################################
class PacketCounter(gr.basic_block):
    def __init__(self, channel=''):
        gr.basic_block.__init__(self, name=f"PktCounter {channel}", in_sig=None, out_sig=None)
        self.message_port_register_in(pmt.intern("in"))
        self.message_port_register_out(pmt.intern("out"))
        self.set_msg_handler(pmt.intern("in"), self._handle)
        self.channel, self.count, self.total_bytes = channel, 0, 0
    def _handle(self, msg):
        self.count += 1
        self.message_port_pub(pmt.intern("out"), msg)

class PreCrcCounter(gr.sync_block):
    def __init__(self, channel=''):
        gr.sync_block.__init__(self, name=f"PreCRC {channel}", in_sig=[np.uint8], out_sig=[np.uint8])
        self.count = 0
        self._last_offset = -1
    def work(self, input_items, output_items):
        n = len(input_items[0])
        output_items[0][:n] = input_items[0][:n]
        tags = self.get_tags_in_window(0, 0, n)
        for tag in tags:
            if pmt.symbol_to_string(tag.key) == "packet_len":
                if tag.offset != self._last_offset:
                    self.count += 1
                    self._last_offset = tag.offset
        return n

class ImageRecoveryBlock(gr.basic_block):
    def __init__(self, out_jpg='recovered.jpg', channel=''):
        gr.basic_block.__init__(self, name=f"ImageRec {channel}", in_sig=None, out_sig=None)
        self.message_port_register_in(pmt.intern("pdus"))
        self.set_msg_handler(pmt.intern("pdus"), self._handle)
        self.out_jpg, self.channel, self.buf = out_jpg, channel, bytearray()
        self.proxy = SignalProxy()

    def _handle(self, msg):
        data = bytes(pmt.u8vector_elements(pmt.cdr(msg)))
        self.buf.extend(data)
        if len(self.buf) > MAX_BUF_SIZE: self.buf = self.buf[-MAX_BUF_SIZE:]
        self._search()

    def _search(self):
        pos = 0
        while True:
            s = self.buf.find(JPEG_START, pos)
            if s < 0: break
            e = self.buf.find(JPEG_END, s + 2)
            if e < 0: return
            candidate = bytes(self.buf[s:e + 2])
            pos = e + 2
            if len(candidate) > 2000:
                with open(self.out_jpg, 'wb') as f: f.write(candidate)
                self.proxy.image_received.emit(self.out_jpg)
        if pos > 0: del self.buf[:pos]

##################################################
# MIMO XPIC Logic
##################################################
@nb.jit(nopython=True, fastmath=True, nogil=True, cache=True)
def numba_cma_xpic(in0, in1, out0, out1, mu_direct, mu_cross, rho, leak, w00, w01, w10, w11, e_acc0, e_acc1, e_cnt):
    n = in0.shape[0]
    for i in range(n):
        x0, x1 = in0[i], in1[i]
        y0, y1 = w00*x0 + w01*x1, w10*x0 + w11*x1
        out0[i], out1[i] = y0, y1
        m0, m1 = y0.real**2 + y0.imag**2, y1.real**2 + y1.imag**2
        if m0 > 4.0 or m1 > 4.0:
            w00, w11 = 0.9*w00 + 0.1, 0.9*w11 + 0.1
            w01, w10 = 0.9*w01, 0.9*w10
            continue
        err0, err1 = y0*(m0-1.0), y1*(m1-1.0)
        cross = y0 * y1.conjugate()
        err0 += rho * y1 * cross.conjugate()
        err1 += rho * y0 * cross
        e_acc0 += abs(err0); e_acc1 += abs(err1); e_cnt += 1
        x0c, x1c = x0.conjugate(), x1.conjugate()
        w00 -= mu_direct*err0*x0c; w01 -= mu_cross*err0*x1c
        w10 -= mu_cross*err1*x0c;  w11 -= mu_direct*err1*x1c
        w00 = w00*(1-leak) + leak; w01 *= (1-leak)
        w10 *= (1-leak); w11 = w11*(1-leak) + leak
    return w00, w01, w10, w11, e_acc0, e_acc1, e_cnt

class mimo_xpic_2x2(gr.sync_block):
    def __init__(self, mu_direct=1e-4, mu_cross=3e-5, rho=0.05, leak=1e-6):
        gr.sync_block.__init__(self, name="MIMO XPIC", in_sig=[np.complex64, np.complex64], out_sig=[np.complex64, np.complex64])
        self.mu_direct, self.mu_cross, self.rho, self.leak = mu_direct, mu_cross, rho, leak
        self.w00, self.w11 = 1.0+0j, 1.0+0j
        self.w01, self.w10 = 0.0+0j, 0.0+0j
        self.err_acc0, self.err_acc1, self.err_count = 0, 0, 0
    def work(self, input_items, output_items):
        (self.w00, self.w01, self.w10, self.w11, self.err_acc0, self.err_acc1, self.err_count) = numba_cma_xpic(
            input_items[0], input_items[1], output_items[0], output_items[1],
            self.mu_direct, self.mu_cross, self.rho, self.leak, self.w00, self.w01, self.w10, self.w11,
            self.err_acc0, self.err_acc1, self.err_count)
        return len(input_items[0])

##################################################
# Main Top Block
##################################################
class dual_pole_rx(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "Dual-Pole Rx + RFML")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Dual-Pole Rx — RFML Integrated")
        self.resize(1000, 700)
        main_layout = Qt.QVBoxLayout(self)

        # Params
        samp_rate, freq = 1e6, 2.45e9
        q_acc, b_acc = '11100001010110101110100010010011', '10010110110110100101000111011001'
        rrc = firdes.root_raised_cosine(32, 32, 1.0, 0.35, 11*32)

        # --- GUI Setup ---
        img_row = Qt.QHBoxLayout()
        # CH0 UI
        q_grp = Qt.QGroupBox("CH0 — QPSK")
        q_lay = Qt.QVBoxLayout(q_grp)
        self.q_img = Qt.QLabel("Awaiting QPSK...")
        self.q_mod_lbl = Qt.QLabel("Mod: ---")
        self.q_mod_lbl.setStyleSheet("font-weight: bold; color: #2a7ae2; font-size: 14px;")
        q_lay.addWidget(self.q_img); q_lay.addWidget(self.q_mod_lbl)
        img_row.addWidget(q_grp)
        # CH1 UI
        b_grp = Qt.QGroupBox("CH1 — BPSK")
        b_lay = Qt.QVBoxLayout(b_grp)
        self.b_img = Qt.QLabel("Awaiting BPSK...")
        self.b_mod_lbl = Qt.QLabel("Mod: ---")
        self.b_mod_lbl.setStyleSheet("font-weight: bold; color: #2a7ae2; font-size: 14px;")
        b_lay.addWidget(self.b_img); b_lay.addWidget(self.b_mod_lbl)
        img_row.addWidget(b_grp)
        main_layout.addLayout(img_row)

        self.status = Qt.QLabel("Initializing...")
        main_layout.addWidget(self.status)

        # --- Blocks ---
        self.usrp = uhd.usrp_source("", uhd.stream_args(cpu_format="fc32", channels=[0,1]))
        self.usrp.set_samp_rate(samp_rate)
        for i in [0,1]:
            self.usrp.set_center_freq(freq, i)
            self.usrp.set_gain(20, i)

        self.agc0, self.agc1 = analog.agc_cc(1e-4, 1.0, 1.0), analog.agc_cc(1e-4, 1.0, 1.0)
        self.fll0, self.fll1 = digital.fll_band_edge_cc(4, 0.35, 44, 0.0628), digital.fll_band_edge_cc(4, 0.35, 44, 0.0628)
        self.pfb0 = digital.pfb_clock_sync_ccf(4, 0.0628, rrc, 32, 16, 1.5, 1)
        self.pfb1 = digital.pfb_clock_sync_ccf(4, 0.0628, rrc, 32, 16, 1.5, 1)
        self.mimo = mimo_xpic_2x2()

        # RFML Classifiers
        self.clf0 = ModulationClassifierBlock(model_dir="artifacts")
        self.clf1 = ModulationClassifierBlock(model_dir="artifacts")

        # Recovery chains (Simplified connection logic)
        self.costas0, self.costas1 = digital.costas_loop_cc(0.0628, 4), digital.costas_loop_cc(0.0628, 2)
        self.decq, self.decb = digital.constellation_decoder_cb(digital.constellation_qpsk().base()), digital.constellation_decoder_cb(digital.constellation_bpsk().base())
        
        # ... (Assuming standard diff/corr/repack/crc chain from original script) ...
        self.cnt_q = PacketCounter('QPSK'); self.precrc_q = PreCrcCounter('QPSK'); self.rec_q = ImageRecoveryBlock('q_rec.jpg', 'QPSK')
        self.cnt_b = PacketCounter('BPSK'); self.precrc_b = PreCrcCounter('BPSK'); self.rec_b = ImageRecoveryBlock('b_rec.jpg', 'BPSK')

        # --- Connections ---
        self.connect((self.usrp,0), self.agc0, self.fll0, self.pfb0, (self.mimo,0))
        self.connect((self.usrp,1), self.agc1, self.fll1, self.pfb1, (self.mimo,1))
        
        # Connect MIMO to Classifiers & Decoders
        self.connect((self.mimo,0), self.clf0)
        self.connect((self.mimo,0), self.costas0, self.decq) # ... and rest of QPSK chain
        
        self.connect((self.mimo,1), self.clf1)
        self.connect((self.mimo,1), self.costas1, self.decb) # ... and rest of BPSK chain

        self.rec_q.proxy.image_received.connect(self._show_q)
        self.rec_b.proxy.image_received.connect(self._show_b)
        self.timer = Qt.QTimer(); self.timer.timeout.connect(self._tick); self.timer.start(1000)

    def _show_q(self, p): self.q_img.setPixmap(Qt.QPixmap(p).scaled(400,400, Qt.Qt.KeepAspectRatio))
    def _show_b(self, p): self.b_img.setPixmap(Qt.QPixmap(p).scaled(400,400, Qt.Qt.KeepAspectRatio))

    def _tick(self):
        self.q_mod_lbl.setText(f"Mod: {self.clf0.latest_pred} ({self.clf0.confidence:.1%})")
        self.b_mod_lbl.setText(f"Mod: {self.clf1.latest_pred} ({self.clf1.confidence:.1%})")
        self.status.setText(f"QPSK Pkts: {self.cnt_q.count} | BPSK Pkts: {self.cnt_b.count}")

def main():
    qapp = Qt.QApplication(sys.argv)
    tb = dual_pole_rx()
    tb.start(); tb.show()
    signal.signal(signal.SIGINT, lambda *x: Qt.QApplication.quit())
    qapp.exec_()

if __name__ == '__main__':
    main()