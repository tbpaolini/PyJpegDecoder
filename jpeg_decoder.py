import numpy as np
from collections import deque

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

# Restart markers
RST = tuple(bytes.fromhex(hex(marker)[2:]) for marker in range(0xFFD0, 0xFFD8))

class JpegDecoder():

    def __init__(self, file) -> None:

        # Open file
        with open(file, "rb") as image:
            self.raw_file = image.read()
        self.file_size = len(self.raw_file)     # Size (in bytes) of the file
        
        # Check if file is a JPEG image
        # (The file needs to start with bytes 'FFD8FF')
        if not self.raw_file.startswith(SOI + b"\xFF"):
            raise NotJpeg("File is not a JPEG image.")

        # Handlers for the markers
        self.handlers = {
            SOI: None,
            SOF0: self.start_of_frame,
            SOF2: self.start_of_frame,
            DHT: self.define_huffman_table,
            DQT: self.define_quantization_table,
            DRI: self.define_restart_interval,
            SOS: self.start_of_scan,
            DNL: self.define_number_of_lines,
            EOI: self.end_of_image,
        }

        # Initialize decoding paramenters
        self.file_header = 2            # Offset (in bytes, 0-index) from the beginning of the file
        self.scan_finished = False      # If the 'end of image' marker has been reached
        self.scan_mode = None           # Supported modes: 'baseline_dct' or 'progressive_dct'
        self.image_width = 0            # Width in pixels of the image
        self.image_height = 0           # Height in pixels of the image
        self.color_component = {}       # Hold each color component and its respective paramenters
        self.huffman_table = {}         # Hold all huffman tables
        self.quantization_table = {}    # Hold all quantization tables
        self.restart_interval = 0       # How many MCU's before each restart marker

        # Loop to find and process the supported file segments
        while not self.scan_finished:
            try:
                current_byte = self.raw_file[self.file_header]
            except IndexError:
                break

            if (current_byte == 0xFF):
                my_marker = self.raw_file[self.file_header : self.file_header+2]
                self.file_header += 2

                if (my_marker != b"\xFF\x00") and (my_marker not in RST):
                    my_handler = self.handlers.get(my_marker)
                    my_size = bytes_to_uint(self.raw_file[self.file_header : self.file_header+2]) - 2
                    self.file_header += 2

                    if my_handler is not None:
                        my_data = self.raw_file[self.file_header : self.file_header+my_size]
                        my_handler(my_data)
                    
                    self.file_header += my_size
            
            else:
                self.file_header += 1


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
        self.scan_finished = True


# ----------------------------------------------------------------------------
# Helper functions

def bytes_to_uint(bytes_obj:bytes) -> int:
    return int.from_bytes(bytes_obj, byteorder="big", signed=False)


# ----------------------------------------------------------------------------
# Decoder exceptions

class JpegError(Exception):
    """Parent of all other exceptions of this decoder."""


class NotJpeg(JpegError):
    """File is not a JPEG image."""


# ----------------------------------------------------------------------------
# Run script

if __name__ == "__main__":
    pass