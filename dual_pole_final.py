#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-Pole Receiver — QPSK (ch0) + BPSK (ch1)
All QPSK parameters match the proven working GRC at 3 GHz exactly.

This script addresses every known failure mode:
  1. Dual-channel USRP — ch1 has a null sink fallback so the scheduler
     always drains both channels even if BPSK isn't decoding
  2. Image recovery — im.load() for strict validation, atomic file
     writes so the GUI never reads a half-written JPEG
  3. Parameters — every value copied verbatim from the working GRC
  4. Diagnostics — terminal output at every stage so you can see
     exactly where the pipeline breaks

Signal flow:
  USRP ch0 -> AGC -> FLL -> PFB(osps=2) -> Costas(4) -> QPSK decode
  USRP ch1 -> AGC -> FLL -> PFB(osps=2) -> Costas(2) -> BPSK decode
"""

import os
import sys
import signal
import io
import tempfile
import numpy as np
import pmt
from PyQt5 import Qt
from PyQt5.QtCore import pyqtSignal, QObject
from gnuradio import analog, blocks, digital, gr, uhd, pdu
from gnuradio.filter import firdes

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARN] PIL/Pillow not installed — image validation will be weak")

JPEG_START = b"\xFF\xD8"
JPEG_END   = b"\xFF\xD9"
MAX_BUF_SIZE = 2_000_000


class SignalProxy(QObject):
    image_received = pyqtSignal(str)


##################################################
# Packet Counter — sits between CRC and PDU to
# count how many packets survive CRC validation.
# If this count stays at 0, the DSP chain isn't
# locking. If it climbs but no images appear, the
# image recovery is the problem.
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
# Main
##################################################
class dual_pole_rx(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "Dual-Pole Rx",
                              catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Dual-Pole Receiver — 3.0 GHz")
        self.resize(950, 580)

        main_layout = Qt.QVBoxLayout(self)

        ##################################################
        # Variables — VERBATIM from working GRC
        #
        # These values produce a working QPSK decode on the
        # B210. Do not optimize, round, or "improve" them.
        ##################################################
        sps         = 4
        samp_rate   = 1e6
        freq        = 3e9
        excess_bw   = 0.35

        rx_gain_ch0 = 20          # from working GRC gain0
        rx_gain_ch1 = 20          # BPSK — start same as QPSK

        # access_code from working GRC: '11100001010110101110100010010011'
        qpsk_access = '11100001010110101110100010010011'
        # BPSK uses a DIFFERENT code to prevent cross-channel false sync
        bpsk_access = '10010110110110100101000111011001'

        # RRC taps — the EXACT expression from the working GRC:
        #   firdes.root_raised_cosine(32, 32, 1.0/1.0, 0.35, 11*32)
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

        self.status = Qt.QLabel("Initializing...")
        self.status.setStyleSheet("font-size: 11px; color: #888; padding: 4px;")
        main_layout.addWidget(self.status)

        ##################################################
        # USRP — 2-channel B210
        #
        # CRITICAL: even if only QPSK Tx is running, the
        # B210 must drain both channels or the scheduler
        # stalls ch0. Ch1 data goes through the full DSP
        # chain anyway — if no BPSK Tx is running, ch1
        # just decodes noise and the image recovery block
        # rejects everything (which is correct behavior).
        ##################################################

        self.usrp = uhd.usrp_source(
            ",".join(("", '')),
            uhd.stream_args(cpu_format="fc32", args='',
                            channels=list(range(0, 2))),
        )
        self.usrp.set_samp_rate(samp_rate)

        # CH0 — exact from working GRC
        self.usrp.set_center_freq(freq, 0)
        self.usrp.set_antenna("RX2", 0)        # GRC uses RX2, not TX/RX
        self.usrp.set_gain(rx_gain_ch0, 0)

        # CH1
        self.usrp.set_center_freq(freq, 1)
        self.usrp.set_antenna("RX2", 1)
        self.usrp.set_gain(rx_gain_ch1, 1)

        print(f"[USRP] 2-ch mode, {samp_rate/1e6} MS/s, "
              f"freq={freq/1e9} GHz, "
              f"gain=[{rx_gain_ch0}, {rx_gain_ch1}]")

        ##################################################
        # AGC — exact from working GRC
        # rate=1e-4, ref=1.0, gain=1.0, max_gain=4000
        ##################################################

        self.agc_0 = analog.agc_cc(1e-4, 1.0, 1.0, 4000)
        self.agc_1 = analog.agc_cc(1e-4, 1.0, 1.0, 4000)

        ##################################################
        # FLL — exact from working GRC
        # sps=4, excess_bw=0.35, filter_size=44, w=0.0628
        # BOTH channels identical
        ##################################################

        self.fll_0 = digital.fll_band_edge_cc(sps, excess_bw, 44, 0.0628)
        self.fll_1 = digital.fll_band_edge_cc(sps, excess_bw, 44, 0.0628)

        ##################################################
        # PFB Clock Sync — exact from working GRC
        # sps=4, loop_bw=0.0628, taps=rrc_taps,
        # nfilts=32, init_phase=16, max_dev=1.5, osps=2
        # BOTH channels identical
        ##################################################

        self.pfb_0 = digital.pfb_clock_sync_ccf(
            sps, 0.0628, rrc_taps, 32, 16, 1.5, 2)
        self.pfb_1 = digital.pfb_clock_sync_ccf(
            sps, 0.0628, rrc_taps, 32, 16, 1.5, 2)

        ##################################################
        # Costas — exact from working GRC
        # QPSK: w=0.0628, order=4
        # BPSK: w=0.0628, order=2
        ##################################################

        self.costas_0 = digital.costas_loop_cc(0.0628, 4, False)
        self.costas_1 = digital.costas_loop_cc(0.0628, 2, False)

        ##################################################
        # QPSK decode — exact from working GRC
        #
        # constellation_qpsk, diff mod=4, unpack k=2,
        # correlate threshold=2, repack 1->8, CRC32 check
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

        # Packet counter — shows CRC-valid packets in terminal
        self.cnt_q   = PacketCounter(channel='QPSK')

        self.rec_q   = ImageRecoveryBlock(
                           out_jpg='qpsk_recovered.jpg', channel='QPSK')

        ##################################################
        # BPSK decode
        #
        # Same structure as QPSK but:
        #   - constellation_bpsk
        #   - diff mod=2
        #   - no unpack_k_bits (BPSK = 1 bit/symbol,
        #     decoder already outputs 0 or 1)
        #   - correlate threshold=2
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

        self.rec_b   = ImageRecoveryBlock(
                           out_jpg='bpsk_recovered.jpg', channel='BPSK')

        ##################################################
        # Null sink on ch1 — INSURANCE
        #
        # Even though ch1 goes through the full DSP chain,
        # if any block stalls (e.g. correlate finds no
        # matches and stops pulling), the USRP scheduler
        # can back up on ch1 and starve ch0. The null sink
        # on the Costas output guarantees ch1 is always
        # drained regardless of what happens downstream.
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
        self.connect((self.pfb_0, 0), (self.costas_0, 0))

        self.connect((self.costas_0, 0), (self.dec_q, 0))
        self.connect((self.dec_q, 0),    (self.diff_q, 0))
        self.connect((self.diff_q, 0),   (self.unp_q, 0))
        self.connect((self.unp_q, 0),    (self.corr_q, 0))
        self.connect((self.corr_q, 0),   (self.rep_q, 0))
        self.connect((self.rep_q, 0),    (self.crc_q, 0))
        self.connect((self.crc_q, 0),    (self.pdu_q, 0))
        self.msg_connect((self.pdu_q, 'pdus'), (self.cnt_q, 'in'))
        self.msg_connect((self.cnt_q, 'out'),  (self.rec_q, 'pdus'))

        # === CH1: BPSK ===
        self.connect((self.usrp, 1), (self.agc_1, 0))
        self.connect((self.agc_1, 0), (self.fll_1, 0))
        self.connect((self.fll_1, 0), (self.pfb_1, 0))
        self.connect((self.pfb_1, 0), (self.costas_1, 0))

        # Null sink guarantees ch1 always drains
        self.connect((self.costas_1, 0), (self.null_1, 0))

        # BPSK decode chain (runs in parallel with null sink)
        self.connect((self.costas_1, 0), (self.dec_b, 0))
        self.connect((self.dec_b, 0),    (self.diff_b, 0))
        self.connect((self.diff_b, 0),   (self.corr_b, 0))
        self.connect((self.corr_b, 0),   (self.rep_b, 0))
        self.connect((self.rep_b, 0),    (self.crc_b, 0))
        self.connect((self.crc_b, 0),    (self.pdu_b, 0))
        self.msg_connect((self.pdu_b, 'pdus'), (self.cnt_b, 'in'))
        self.msg_connect((self.cnt_b, 'out'),  (self.rec_b, 'pdus'))

        print("[INIT] Flowgraph built. Starting...")

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
        qp = self.rec_q.pkt_count
        qv = self.rec_q.valid_count
        qr = self.rec_q.reject_count
        bp = self.rec_b.pkt_count
        bv = self.rec_b.valid_count
        br = self.rec_b.reject_count
        qc = self.cnt_q.count
        bc = self.cnt_b.count
        self.status.setText(
            f"QPSK: {qc} CRC-ok → {qp} to recovery → "
            f"{qv} valid / {qr} rejected  |  "
            f"BPSK: {bc} CRC-ok → {bp} to recovery → "
            f"{bv} valid / {br} rejected")

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

    print("[RUN] Flowgraph running. Watch terminal for packet counts.")
    print("[RUN] If QPSK CRC-ok count stays 0, the DSP chain isn't locking.")
    print("[RUN] If CRC-ok climbs but valid images stays 0, recovery is failing.")

    qapp.exec_()


if __name__ == '__main__':
    main()
