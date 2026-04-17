#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-Pole Receiver — Post-Costas IQ Capture (headless)

Signal flow (unchanged up through Costas):
  USRP ch0 -> AGC -> FLL -> PFB(osps=2) -> MIMO XPIC -> Costas(4) -> file sink
  USRP ch1 -> AGC -> FLL -> PFB(osps=2) -> MIMO XPIC -> Costas(2) -> file sink

Writes interleaved complex64 (I,Q float32 pairs) to:
  ch0_postcostas.iq   (QPSK branch)
  ch1_postcostas.iq   (BPSK branch)

No UI, no RFML, no packet decode, no image recovery.
Run until Ctrl-C.
"""

import sys
import signal
import numpy as np
import numba as nb
from gnuradio import analog, blocks, digital, gr, uhd
from gnuradio.filter import firdes


# =============================================================================
# Output files
# =============================================================================
OUT_CH0 = "ch0_postcostas.iq"   # QPSK, post-Costas
OUT_CH1 = "ch1_postcostas.iq"   # BPSK, post-Costas


##################################################
# NUMBA-Compiled CMA XPIC
# (unchanged from original — needed because it sits
#  between PFB and Costas in the signal chain)
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
        w01 -= mu_cross * err0 * x1c
        w10 -= mu_cross * err1 * x0c
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


##################################################
# Flowgraph
##################################################
class dual_pole_rx_capture(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "Dual-Pole Rx Capture",
                              catch_exceptions=True)

        # --- parameters (unchanged from original) ---
        sps         = 4
        samp_rate   = 1e6
        freq        = 2.45e9
        excess_bw   = 0.35
        rx_gain_ch0 = 20
        rx_gain_ch1 = 20

        rrc_taps = firdes.root_raised_cosine(32, 32, 1.0 / 1.0, 0.35, 11 * 32)

        # --- USRP: 2-ch B210 ---
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

        # --- AGC ---
        self.agc_0 = analog.agc_cc(1e-4, 1.0, 1.0, 4000)
        self.agc_1 = analog.agc_cc(1e-4, 1.0, 1.0, 4000)

        # --- FLL ---
        self.fll_0 = digital.fll_band_edge_cc(sps, excess_bw, 44, 0.0628)
        self.fll_1 = digital.fll_band_edge_cc(sps, excess_bw, 44, 0.0628)

        # --- PFB Clock Sync ---
        self.pfb_0 = digital.pfb_clock_sync_ccf(
            sps, 0.0628, rrc_taps, 32, 16, 1.5, 1)
        self.pfb_1 = digital.pfb_clock_sync_ccf(
            sps, 0.0628, rrc_taps, 32, 16, 1.5, 1)

        # --- MIMO XPIC ---
        self.mimo = mimo_xpic_2x2(
            mu_direct=1e-4,
            mu_cross=3e-5,
            rho=0.05,
            leak=1e-6
        )

        # --- Costas ---
        self.costas_0 = digital.costas_loop_cc(0.0628, 4, False)
        self.costas_1 = digital.costas_loop_cc(0.0628, 2, False)

        # --- File sinks (interleaved complex64 -> standard .iq format) ---
        self.sink_0 = blocks.file_sink(
            gr.sizeof_gr_complex, OUT_CH0, False)
        self.sink_0.set_unbuffered(False)

        self.sink_1 = blocks.file_sink(
            gr.sizeof_gr_complex, OUT_CH1, False)
        self.sink_1.set_unbuffered(False)

        # --- Connections ---
        # CH0
        self.connect((self.usrp, 0),    (self.agc_0, 0))
        self.connect((self.agc_0, 0),   (self.fll_0, 0))
        self.connect((self.fll_0, 0),   (self.pfb_0, 0))
        self.connect((self.pfb_0, 0),   (self.mimo, 0))
        self.connect((self.mimo, 0),    (self.costas_0, 0))
        self.connect((self.costas_0, 0), (self.sink_0, 0))

        # CH1
        self.connect((self.usrp, 1),    (self.agc_1, 0))
        self.connect((self.agc_1, 0),   (self.fll_1, 0))
        self.connect((self.fll_1, 0),   (self.pfb_1, 0))
        self.connect((self.pfb_1, 0),   (self.mimo, 1))
        self.connect((self.mimo, 1),    (self.costas_1, 0))
        self.connect((self.costas_1, 0), (self.sink_1, 0))

        print(f"[SINK] CH0 (QPSK) -> {OUT_CH0}")
        print(f"[SINK] CH1 (BPSK) -> {OUT_CH1}")
        print("[INIT] Flowgraph built. Starting...")


def main():
    tb = dual_pole_rx_capture()

    def sig_handler(sig=None, frame=None):
        print("\n[STOP] Stopping flowgraph...")
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    print("[RUN] Capturing post-Costas IQ. Ctrl-C to stop.")
    tb.wait()


if __name__ == '__main__':
    main()