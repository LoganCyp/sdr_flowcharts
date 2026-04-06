#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-Pole MIMO Front-End Diagnostic
Runs: USRP -> AGC -> FLL -> PFB(1sps) -> MIMO XPIC -> Costas
Displays live constellation plots for CH0 (QPSK) and CH1 (BPSK).
No bit decoding — purely for verifying RF lock and constellation quality.
"""

import sys
import signal
import numpy as np
import numba as nb
from PyQt5 import Qt
from PyQt5.QtCore import QTimer
from gnuradio import analog, blocks, digital, gr, uhd
from gnuradio.filter import firdes

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


##################################################
# NUMBA MIMO XPIC (identical to production)
##################################################
@nb.jit(nopython=True, fastmath=True, nogil=True, cache=True)
def numba_cma_xpic(in0, in1, out0, out1, mu, rho, w00, w01, w10, w11):
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

        if mag_sq0 > 10.0 or mag_sq1 > 10.0:
            w00 = 1.0 + 0j
            w01 = 0.0 + 0j
            w10 = 0.0 + 0j
            w11 = 1.0 + 0j
            continue

        err0 = y0 * (mag_sq0 - 1.0)
        err1 = y1 * (mag_sq1 - 1.0)

        cross = y0 * y1.conjugate()
        err0 += rho * y1 * cross.conjugate()
        err1 += rho * y0 * cross

        x0c = x0.conjugate()
        x1c = x1.conjugate()

        w00 -= mu * err0 * x0c
        w01 -= mu * err0 * x1c
        w10 -= mu * err1 * x0c
        w11 -= mu * err1 * x1c

    return w00, w01, w10, w11


class mimo_xpic_2x2(gr.sync_block):
    def __init__(self, mu=1e-4, rho=0.05):
        gr.sync_block.__init__(self, name="MIMO XPIC 2x2",
                               in_sig=[np.complex64, np.complex64],
                               out_sig=[np.complex64, np.complex64])
        self.mu = mu
        self.rho = rho
        self.w00 = 1.0 + 0j
        self.w01 = 0.0 + 0j
        self.w10 = 0.0 + 0j
        self.w11 = 1.0 + 0j

    def work(self, input_items, output_items):
        self.w00, self.w01, self.w10, self.w11 = numba_cma_xpic(
            input_items[0], input_items[1],
            output_items[0], output_items[1],
            self.mu, self.rho,
            self.w00, self.w01, self.w10, self.w11
        )
        return len(input_items[0])


##################################################
# Ring buffer sink — stores last N complex samples
##################################################
class ring_sink(gr.sync_block):
    def __init__(self, buf_size=4096):
        gr.sync_block.__init__(self, name="ring_sink",
                               in_sig=[np.complex64], out_sig=None)
        self.buf = np.zeros(buf_size, dtype=np.complex64)
        self.buf_size = buf_size
        self.write_ptr = 0
        self.filled = False

    def work(self, input_items, output_items):
        data = input_items[0]
        n = len(data)

        if n >= self.buf_size:
            self.buf[:] = data[-self.buf_size:]
            self.write_ptr = 0
            self.filled = True
        else:
            end = self.write_ptr + n
            if end <= self.buf_size:
                self.buf[self.write_ptr:end] = data
            else:
                first = self.buf_size - self.write_ptr
                self.buf[self.write_ptr:] = data[:first]
                self.buf[:n - first] = data[first:]
                self.filled = True
            self.write_ptr = end % self.buf_size

        return n

    def get_data(self):
        if self.filled:
            return np.copy(self.buf)
        else:
            return np.copy(self.buf[:self.write_ptr])


##################################################
# Matplotlib constellation canvas
##################################################
class ConstellationCanvas(FigureCanvasQTAgg):
    def __init__(self, title="Constellation", parent=None):
        self.fig = Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.title = title
        self._setup_axes()

    def _setup_axes(self):
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linewidth=0.5)
        self.ax.axhline(0, color='gray', linewidth=0.5)
        self.ax.axvline(0, color='gray', linewidth=0.5)
        self.ax.set_title(self.title, fontsize=14, fontweight='bold')
        self.ax.tick_params(labelsize=8)

    def update_plot(self, data):
        self.ax.clear()
        self._setup_axes()

        if len(data) > 0:
            self.ax.scatter(data.real, data.imag,
                            s=1, c='blue', alpha=0.6,
                            edgecolors='none', rasterized=True)

        self.fig.canvas.draw_idle()


##################################################
# Main window
##################################################
class DiagnosticWindow(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "MIMO Front-End Diagnostic",
                              catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("MIMO Constellation Diagnostic")
        self.resize(1100, 550)

        ##################################################
        # Parameters
        ##################################################
        sps       = 4
        nfilts    = 32
        samp_rate = 1e6
        freq      = 2.45e9
        excess_bw = 0.35
        rx_gain   = 50

        ##################################################
        # GUI — two constellation plots
        ##################################################
        layout = Qt.QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.canvas_ch0 = ConstellationCanvas("CH0 — QPSK (Post Costas)")
        self.canvas_ch1 = ConstellationCanvas("CH1 — BPSK (Post Costas)")

        layout.addWidget(self.canvas_ch0)
        layout.addWidget(self.canvas_ch1)

        ##################################################
        # USRP Source — 2 channels
        ##################################################
        self.usrp = uhd.usrp_source(
            ",".join(("", '')),
            uhd.stream_args(cpu_format="fc32", args='',
                            channels=list(range(0, 2))),
        )
        self.usrp.set_samp_rate(samp_rate)

        self.usrp.set_center_freq(freq, 0)
        self.usrp.set_antenna("RX2", 0)
        self.usrp.set_gain(rx_gain, 0)

        self.usrp.set_center_freq(freq, 1)
        self.usrp.set_antenna("RX2", 1)
        self.usrp.set_gain(rx_gain, 1)

        ##################################################
        # AGC
        ##################################################
        self.agc_0 = analog.agc_cc(1e-4, 1.0, 1.0, 65536)
        self.agc_1 = analog.agc_cc(1e-4, 1.0, 1.0, 65536)

        ##################################################
        # FLL Band Edge
        ##################################################
        self.fll_0 = digital.fll_band_edge_cc(sps, excess_bw, 44, 0.0125)
        self.fll_1 = digital.fll_band_edge_cc(sps, excess_bw, 44, 0.0125)

        ##################################################
        # PFB Clock Sync — osps=1
        ##################################################
        rrc_taps = firdes.root_raised_cosine(
            nfilts, nfilts, 1.0 / float(sps), excess_bw,
            11 * sps * nfilts
        )

        self.pfb_0 = digital.pfb_clock_sync_ccf(
            sps, 0.02, rrc_taps, nfilts, nfilts // 2, 1.5, 1)
        self.pfb_1 = digital.pfb_clock_sync_ccf(
            sps, 0.02, rrc_taps, nfilts, nfilts // 2, 1.5, 1)

        ##################################################
        # MIMO XPIC — at 1 sps
        ##################################################
        self.mimo = mimo_xpic_2x2(mu=1e-4, rho=0.05)

        ##################################################
        # Costas Loops
        ##################################################
        self.costas_0 = digital.costas_loop_cc(0.025, 4, False)
        self.costas_1 = digital.costas_loop_cc(0.02,  2, False)

        ##################################################
        # Ring buffer sinks — grab samples for plotting
        ##################################################
        self.sink_0 = ring_sink(buf_size=4096)
        self.sink_1 = ring_sink(buf_size=4096)

        ##################################################
        # Connections
        # USRP -> AGC -> FLL -> PFB(1sps) -> MIMO -> Costas -> sink
        ##################################################
        self.connect((self.usrp, 0), (self.agc_0, 0))
        self.connect((self.usrp, 1), (self.agc_1, 0))

        self.connect((self.agc_0, 0), (self.fll_0, 0))
        self.connect((self.agc_1, 0), (self.fll_1, 0))

        self.connect((self.fll_0, 0), (self.pfb_0, 0))
        self.connect((self.fll_1, 0), (self.pfb_1, 0))

        self.connect((self.pfb_0, 0), (self.mimo, 0))
        self.connect((self.pfb_1, 0), (self.mimo, 1))

        self.connect((self.mimo, 0), (self.costas_0, 0))
        self.connect((self.mimo, 1), (self.costas_1, 0))

        self.connect((self.costas_0, 0), (self.sink_0, 0))
        self.connect((self.costas_1, 0), (self.sink_1, 0))

        ##################################################
        # Refresh timer — update plots every 100 ms
        ##################################################
        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh)
        self.timer.start(100)

    def _refresh(self):
        self.canvas_ch0.update_plot(self.sink_0.get_data())
        self.canvas_ch1.update_plot(self.sink_1.get_data())

    def closeEvent(self, event):
        self.timer.stop()
        self.stop()
        self.wait()
        event.accept()


def main():
    qapp = Qt.QApplication(sys.argv)
    tb = DiagnosticWindow()
    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.timer.stop()
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    qapp.exec_()


if __name__ == '__main__':
    main()