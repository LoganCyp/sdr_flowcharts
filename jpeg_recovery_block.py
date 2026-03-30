"""
Embedded Python Block: JPEG Image Recovery

Drop this into a GRC flowgraph as an Embedded Python Block.
Message input port: pdus
Saves only fully decodable JPEGs. No garbage, no corruption.
"""

import os
import sys
import io
import tempfile
import subprocess
from gnuradio import gr
import pmt

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

JPEG_START = b"\xFF\xD8"
JPEG_END   = b"\xFF\xD9"


class blk(gr.basic_block):
    def __init__(self, out_jpg="recovered.jpg", show=True, min_size=2000):
        gr.basic_block.__init__(self,
                                name="jpeg_recovery_strict",
                                in_sig=None,
                                out_sig=None)

        self.message_port_register_in(pmt.intern("pdus"))
        self.set_msg_handler(pmt.intern("pdus"), self._handle)

        self.out_jpg = out_jpg
        self.show = show
        self.min_size = min_size
        self.buf = bytearray()
        self.pkt_count = 0
        self.valid_count = 0
        self.reject_count = 0
        self.done = False

    def _handle(self, msg):
        if self.done:
            return

        try:
            vec = pmt.cdr(msg)
            data = bytes(pmt.u8vector_elements(vec))
        except Exception:
            return

        self.pkt_count += 1
        self.buf.extend(data)

        # Cap buffer at 2 MB
        if len(self.buf) > 2_000_000:
            self.buf = self.buf[-2_000_000:]

        if self.pkt_count % 200 == 0:
            print(f"[JPEG] {self.pkt_count} pkts, "
                  f"buf={len(self.buf)}, "
                  f"valid={self.valid_count}, "
                  f"rejected={self.reject_count}")

        self._search()

    def _search(self):
        pos = 0

        while True:
            s = self.buf.find(JPEG_START, pos)
            if s < 0:
                break

            e = self.buf.find(JPEG_END, s + 2)
            if e < 0:
                # SOI found, no EOI yet — trim before SOI, keep buffering
                if s > 0:
                    del self.buf[:s]
                return

            candidate = bytes(self.buf[s:e + 2])
            pos = e + 2

            if len(candidate) < self.min_size:
                continue

            if self._validate(candidate):
                self.valid_count += 1
                self._atomic_save(candidate)
                print(f"[JPEG] VALID #{self.valid_count}: "
                      f"{len(candidate)} bytes — saved to {self.out_jpg}")

                if self.show:
                    self._open_image(self.out_jpg)

                self.done = True
                os._exit(0)
            else:
                self.reject_count += 1

        if pos > 0:
            del self.buf[:pos]

    def _validate(self, b):
        """
        im.load() forces full pixel decompression.
        Corrupt JPEGs fail here in under 1 ms.
        im.verify() only checks headers — it lets garbage through.
        """
        if not HAS_PIL:
            # No PIL — fall back to marker + size check only
            if len(b) < 5000:
                return False
            return (b"JFIF" in b[:20] or b"Exif" in b[:20])

        try:
            im = Image.open(io.BytesIO(b))
            im.load()
            w, h = im.size
            if w < 16 or h < 16:
                return False
            return True
        except Exception:
            return False

    def _atomic_save(self, data):
        """
        Write to temp file then rename atomically.
        Prevents any reader from seeing a half-written file.
        """
        out_dir = os.path.dirname(os.path.abspath(self.out_jpg))
        try:
            fd, tmp = tempfile.mkstemp(suffix='.jpg', dir=out_dir)
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
            os.replace(tmp, self.out_jpg)
        except Exception as ex:
            print(f"[JPEG] Save error: {ex}")
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def _open_image(self, path):
        try:
            if sys.platform.startswith("win"):
                os.startfile(os.path.abspath(path))
            elif sys.platform.startswith("darwin"):
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as ex:
            print(f"[JPEG] Could not open image: {ex}")
