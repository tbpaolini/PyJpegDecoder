import numpy as np
from collections import deque

class JpegDecoder():
    
    # JPEG markers (for our supported segments)
    SOI  = bytes.fromhex("FFD8")    # Start of image
    SOF0 = bytes.fromhex("FFC0")    # Start of frame (Baseline DCT)
    SOF2 = bytes.fromhex("FFC2")    # Start of frame (Progressive DCT)
    DHT  = bytes.fromhex("FFC4")    # Define Huffman table
    DQT  = bytes.fromhex("FFDB")    # Define quantization table
    DRI  = bytes.fromhex("FFDD")    # Define restart interval
    SOS  = bytes.fromhex("FFDA")    # Start of scan
    DNL  = bytes.fromhex("FFDC")    # Define number of lines
    EOI  = bytes.fromhex("FFD9")    # End of image

    def __init__(self) -> None:

        # Handlers for the markers
        self.handlers = {
            self.SOI: None,
            self.SOF0: self.start_of_frame,
            self.SOF2: self.start_of_frame,
            self.DHT: self.define_huffman_table,
            self.DQT: self.define_quantization_table,
            self.DRI: self.define_restart_interval,
            self.SOS: self.start_of_scan,
            self.DNL: self.define_number_of_lines,
            self.EOI: None,
        }

    def start_of_frame(self):
        pass

    def define_huffman_table(self):
        pass

    def define_quantization_table(self):
        pass

    def define_restart_interval(self):
        pass

    def define_number_of_lines(self):
        pass

    def start_of_scan(self):
        pass

    def baseline_dct_scan(self):
        pass

    def progressive_dct_scan(self):
        pass

    def end_of_image(self):
        pass


class JpegError(Exception):
    pass


if __name__ == "__main__":
    JpegDecoder("olá")
    pass