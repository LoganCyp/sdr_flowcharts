#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5 import Qt
from gnuradio import qtgui, blocks, digital, gr, uhd
import pmt
import sys
import signal

class dual_pole_tx_b210(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Dual Pole Transmitter", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Dual Pole TX (B210) - QPSK & BPSK")
        self.resize(900, 500)
        
        # Main Layout: Side-by-side columns
        self.main_layout = Qt.QHBoxLayout(self)

        ##################################################
        # Variables
        ##################################################
        self.sps = 4
        self.samp_rate = 1e6
        self.excess_bw = 0.35
        self.center_freq = 5.8e9
        
        # Channel 0 (QPSK) Variables
        self.qpsk_access_code = '11100001010110101110100010010011'
        self.qpsk_image_path = '/home/sdr_caci1/Desktop/Sample Images/caci_bg.jpeg'
        self.qpsk_gain = 78

        # Channel 1 (BPSK) Variables
        self.bpsk_access_code = '10010110110110100101000111011001'
        self.bpsk_image_path = '/home/sdr_caci1/Desktop/Sample Images/lichtenstein-castle.jpg'
        self.bpsk_gain = 72

        ##################################################
        # GUI Setup
        ##################################################
        
        # --- QPSK Control Group (Channel 0) ---
        self.qpsk_group = Qt.QGroupBox("Channel 0: QPSK Transmitter")
        self.qpsk_layout = Qt.QVBoxLayout()
        self.qpsk_group.setLayout(self.qpsk_layout)

        # Image preview label for QPSK
        self.qpsk_preview = Qt.QLabel()
        self.qpsk_preview.setFixedHeight(180)
        self.qpsk_preview.setAlignment(Qt.Qt.AlignCenter)
        self.qpsk_preview.setStyleSheet(
            "background-color: #1a1a2e; border: 1px solid #444; color: #888;"
        )
        self.qpsk_preview.setText("No image selected")
        self.qpsk_layout.addWidget(self.qpsk_preview)
        self._set_preview(self.qpsk_preview, self.qpsk_image_path)

        self.qpsk_file_btn = Qt.QPushButton("Select QPSK Image...")
        self.qpsk_file_btn.clicked.connect(self.open_qpsk_file_dialog)
        self.qpsk_layout.addWidget(self.qpsk_file_btn)
        
        self.qpsk_path_display = Qt.QLineEdit(self.qpsk_image_path)
        self.qpsk_path_display.setReadOnly(True)
        self.qpsk_layout.addWidget(self.qpsk_path_display)
        
        self.qpsk_gain_label = Qt.QLabel(f"<b>QPSK Gain:</b> {self.qpsk_gain} dB")
        self.qpsk_layout.addWidget(self.qpsk_gain_label)
        
        self.qpsk_gain_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self.qpsk_gain_slider.setRange(0, 90)
        self.qpsk_gain_slider.setValue(self.qpsk_gain)
        self.qpsk_gain_slider.valueChanged.connect(self.set_qpsk_gain)
        self.qpsk_layout.addWidget(self.qpsk_gain_slider)
        
        self.main_layout.addWidget(self.qpsk_group)

        # --- BPSK Control Group (Channel 1) ---
        self.bpsk_group = Qt.QGroupBox("Channel 1: BPSK Transmitter")
        self.bpsk_layout = Qt.QVBoxLayout()
        self.bpsk_group.setLayout(self.bpsk_layout)

        # Image preview label for BPSK
        self.bpsk_preview = Qt.QLabel()
        self.bpsk_preview.setFixedHeight(180)
        self.bpsk_preview.setAlignment(Qt.Qt.AlignCenter)
        self.bpsk_preview.setStyleSheet(
            "background-color: #1a1a2e; border: 1px solid #444; color: #888;"
        )
        self.bpsk_preview.setText("No image selected")
        self.bpsk_layout.addWidget(self.bpsk_preview)
        self._set_preview(self.bpsk_preview, self.bpsk_image_path)

        self.bpsk_file_btn = Qt.QPushButton("Select BPSK Image...")
        self.bpsk_file_btn.clicked.connect(self.open_bpsk_file_dialog)
        self.bpsk_layout.addWidget(self.bpsk_file_btn)
        
        self.bpsk_path_display = Qt.QLineEdit(self.bpsk_image_path)
        self.bpsk_path_display.setReadOnly(True)
        self.bpsk_layout.addWidget(self.bpsk_path_display)
        
        self.bpsk_gain_label = Qt.QLabel(f"<b>BPSK Gain:</b> {self.bpsk_gain} dB")
        self.bpsk_layout.addWidget(self.bpsk_gain_label)
        
        self.bpsk_gain_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self.bpsk_gain_slider.setRange(0, 90)
        self.bpsk_gain_slider.setValue(self.bpsk_gain)
        self.bpsk_gain_slider.valueChanged.connect(self.set_bpsk_gain)
        self.bpsk_layout.addWidget(self.bpsk_gain_slider)
        
        self.main_layout.addWidget(self.bpsk_group)

        ##################################################
        # Blocks setup
        ##################################################

        # 2-Channel USRP Sink
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(("", '')),
            uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0,2))),
            "",
        )
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        
        # Channel 0 Init
        self.uhd_usrp_sink_0.set_center_freq(self.center_freq, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_gain(self.qpsk_gain, 0)
        
        # Channel 1 Init
        self.uhd_usrp_sink_0.set_center_freq(self.center_freq, 1)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 1)
        self.uhd_usrp_sink_0.set_gain(self.bpsk_gain, 1)

        # --- QPSK Blocks (Channel 0) ---
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_char, self.qpsk_image_path, True)
        self.blocks_stream_to_tagged_stream_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, 128, "packet_len")
        self.digital_crc32_bb_0 = digital.crc32_bb(False, "packet_len", True)
        self.digital_protocol_formatter_bb_0 = digital.protocol_formatter_bb(
            digital.header_format_default(self.qpsk_access_code, 0), "packet_len")
        self.blocks_tagged_stream_mux_0 = blocks.tagged_stream_mux(gr.sizeof_char, "packet_len", 0)
        self.digital_constellation_modulator_0 = digital.generic_mod(
            constellation=digital.constellation_qpsk().base(),
            differential=True,
            samples_per_symbol=self.sps,
            pre_diff_code=True,
            excess_bw=self.excess_bw)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(0.4)

        # --- BPSK Blocks (Channel 1) ---
        self.blocks_file_source_0_0 = blocks.file_source(gr.sizeof_char, self.bpsk_image_path, True)
        self.blocks_stream_to_tagged_stream_0_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, 128, "packet_len")
        self.digital_crc32_bb_0_0 = digital.crc32_bb(False, "packet_len", True)
        self.digital_protocol_formatter_bb_0_0 = digital.protocol_formatter_bb(
            digital.header_format_default(self.bpsk_access_code, 0), "packet_len")
        self.blocks_tagged_stream_mux_0_0 = blocks.tagged_stream_mux(gr.sizeof_char, "packet_len", 0)
        self.digital_constellation_modulator_0_0 = digital.generic_mod(
            constellation=digital.constellation_bpsk().base(),
            differential=True,
            samples_per_symbol=self.sps,
            pre_diff_code=True,
            excess_bw=self.excess_bw)
        self.blocks_multiply_const_vxx_0_0 = blocks.multiply_const_cc(0.8)


        ##################################################
        # Connections
        ##################################################
        
        # QPSK Path Connections -> USRP Channel 0
        self.connect((self.blocks_file_source_0, 0), (self.blocks_stream_to_tagged_stream_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0, 0), (self.digital_crc32_bb_0, 0))
        self.connect((self.digital_crc32_bb_0, 0), (self.digital_protocol_formatter_bb_0, 0))
        self.connect((self.digital_crc32_bb_0, 0), (self.blocks_tagged_stream_mux_0, 1))
        self.connect((self.digital_protocol_formatter_bb_0, 0), (self.blocks_tagged_stream_mux_0, 0))
        self.connect((self.blocks_tagged_stream_mux_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.uhd_usrp_sink_0, 0))

        # BPSK Path Connections -> USRP Channel 1
        self.connect((self.blocks_file_source_0_0, 0), (self.blocks_stream_to_tagged_stream_0_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0_0, 0), (self.digital_crc32_bb_0_0, 0))
        self.connect((self.digital_crc32_bb_0_0, 0), (self.digital_protocol_formatter_bb_0_0, 0))
        self.connect((self.digital_crc32_bb_0_0, 0), (self.blocks_tagged_stream_mux_0_0, 1))
        self.connect((self.digital_protocol_formatter_bb_0_0, 0), (self.blocks_tagged_stream_mux_0_0, 0))
        self.connect((self.blocks_tagged_stream_mux_0_0, 0), (self.digital_constellation_modulator_0_0, 0))
        self.connect((self.digital_constellation_modulator_0_0, 0), (self.blocks_multiply_const_vxx_0_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0_0, 0), (self.uhd_usrp_sink_0, 1))

    ##################################################
    # Helper: load & scale a pixmap into a preview QLabel
    ##################################################

    def _set_preview(self, label: Qt.QLabel, path: str):
        """Load image at `path` and display it scaled into `label`. 
        If the file can't be loaded, show a placeholder message instead."""
        pixmap = Qt.QPixmap(path)
        if pixmap.isNull():
            label.setPixmap(Qt.QPixmap())
            label.setText("Preview unavailable")
        else:
            label.setText("")
            scaled = pixmap.scaled(
                label.width() or 400,
                label.height(),
                Qt.Qt.KeepAspectRatio,
                Qt.Qt.SmoothTransformation,
            )
            label.setPixmap(scaled)

    ##################################################
    # Callbacks
    ##################################################

    def open_qpsk_file_dialog(self):
        filename, _ = Qt.QFileDialog.getOpenFileName(
            self, "Open QPSK Image", "", "Images (*.jpg *.png *.jpeg);;All Files (*)"
        )
        if filename:
            self.qpsk_image_path = filename
            self.qpsk_path_display.setText(filename)
            self.blocks_file_source_0.open(self.qpsk_image_path, True)
            self._set_preview(self.qpsk_preview, self.qpsk_image_path)

    def open_bpsk_file_dialog(self):
        filename, _ = Qt.QFileDialog.getOpenFileName(
            self, "Open BPSK Image", "", "Images (*.jpg *.png *.jpeg);;All Files (*)"
        )
        if filename:
            self.bpsk_image_path = filename
            self.bpsk_path_display.setText(filename)
            self.blocks_file_source_0_0.open(self.bpsk_image_path, True)
            self._set_preview(self.bpsk_preview, self.bpsk_image_path)

    def set_qpsk_gain(self, value):
        self.qpsk_gain = value
        self.qpsk_gain_label.setText(f"<b>QPSK Gain:</b> {self.qpsk_gain} dB")
        self.uhd_usrp_sink_0.set_gain(self.qpsk_gain, 0)

    def set_bpsk_gain(self, value):
        self.bpsk_gain = value
        self.bpsk_gain_label.setText(f"<b>BPSK Gain:</b> {self.bpsk_gain} dB")
        self.uhd_usrp_sink_0.set_gain(self.bpsk_gain, 1)

    def closeEvent(self, event):
        self.stop()
        self.wait()
        event.accept()

def main():
    qapp = Qt.QApplication(sys.argv)
    tb = dual_pole_tx_b210()
    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    qapp.exec_()

if __name__ == '__main__':
    main()
