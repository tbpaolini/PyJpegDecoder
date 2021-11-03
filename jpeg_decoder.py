import numpy as np
from collections import deque, namedtuple
from typing import Callable

from numpy.core.records import array
from numpy.core.shape_base import block

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

# Containers for the parameters of each color component
ColorComponent = namedtuple("ColorComponent", "name vertical_sampling horizontal_sampling quantization_table_id repeat")
HuffmanTable = namedtuple("HuffmanTable", "dc ac")

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
            DHT: self.define_huffman_table,
            DQT: self.define_quantization_table,
            DRI: self.define_restart_interval,
            SOF0: self.start_of_frame,
            SOF2: self.start_of_frame,
            SOS: self.start_of_scan,
            EOI: self.end_of_image,
        }

        # Initialize decoding paramenters
        self.file_header = 2            # Offset (in bytes, 0-index) from the beginning of the file
        self.scan_finished = False      # If the 'end of image' marker has been reached
        self.scan_mode = None           # Supported modes: 'baseline_dct' or 'progressive_dct'
        self.image_width = 0            # Width in pixels of the image
        self.image_height = 0           # Height in pixels of the image
        self.color_components = {}      # Hold each color component and its respective paramenters
        self.huffman_tables = {}        # Hold all huffman tables
        self.quantization_tables = {}   # Hold all quantization tables
        self.restart_interval = 0       # How many MCU's before each restart marker

        # Loop to find and process the supported file segments
        while not self.scan_finished:
            try:
                current_byte = self.raw_file[self.file_header]
            except IndexError:
                del self.raw_file
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
                    else:
                        self.file_header += my_size
            
            else:
                self.file_header += 1

    def start_of_frame(self, data:bytes) -> None:
        data_size = len(data)
        data_header = 0
        
        # Check encoding mode
        mode = self.raw_file[self.file_header-4 : self.file_header-2]
        if mode == SOF0:
            self.scan_mode = "baseline_dct"
        elif mode == SOF2:
            self.scan_mode = "progressive_dct"
        else:
            raise UnsupportedJpeg("Encoding mode not supported. Only 'Baseline DCT' and 'Progressive DCT' are supported.")
        
        # Check sample precision
        # (This is the number of bits used to represent each color value of a pixel)
        precision = data[data_header]
        if precision != 8:
            raise UnsupportedJpeg("Unsupported color depth. Only 8-bit greyscale and 24-bit RGB are supported.")
        data_header += 1
        
        # Get image dimensions
        self.image_height = bytes_to_uint(data[data_header : data_header+2])
        data_header += 2
        """NOTE
        If the heigth is specified as zero here, then it means that the heigth value
        is going to be specidied on the DNL segment after the first scan.
        This is for the case when the final heigth is unknown when the image is
        being created, for example when a scanner is generating the image.
        """

        self.image_width = bytes_to_uint(data[data_header : data_header+2])
        data_header += 2

        if self.image_width == 0:
            raise CorruptedJpeg("Image width cannot be zero.")

        # Check number of color components
        components_amount = data[data_header]
        if components_amount not in (1, 3):
            if components_amount == 4:
                raise UnsupportedJpeg("CMYK color space is not supported. Only RGB and greyscale are supported.")
            else:
                raise UnsupportedJpeg("Unsupported color space. Only RGB and greyscale are supported.")
        data_header += 1

        # Get the color components and their parameters
        components = (
            "Y",    # Luminance
            "Cb",   # Blue chrominance
            "Cr",   # Red chrominance
        )

        try:
            for count, component in enumerate(components, start=1):
                
                # Get the ID of the color component
                my_id = data[data_header]
                data_header += 1

                # Get the horizontal and vertical sampling of the component
                sample = data[data_header]          # This value is 8 bits long
                horizontal_sample = sample >> 4     # Get the first 4 bits of the value
                vertical_sample = sample & 0x0F     # Get the last 4 bits of the value

                data_header += 1

                # Get the quantization table for the component
                my_quantization_table = data[data_header]
                data_header += 1

                # Group the parameters of the component
                my_component = ColorComponent(
                    name = component,                                       # Name of the color component
                    horizontal_sampling = horizontal_sample,                # Amount of pixels sampled in the horizontal
                    vertical_sampling = vertical_sample,                    # # Amount of pixels sampled in the vertical
                    quantization_table_id = my_quantization_table,          # Quantization table selector
                    repeat = horizontal_sample * vertical_sample,           # Amount of times the component repeats during decoding
                )

                # Add the component parameters to the dictionary
                self.color_components.update({my_id: my_component})

                # Have we parsed all components?
                if count == components_amount:
                    break
        
        except IndexError:
            raise CorruptedJpeg("Failed to parse the start of frame.")
        
        # Move the file header to the end of the data segment
        self.file_header += data_size

    def define_huffman_table(self, data:bytes) -> None:
        data_size = len(data)
        data_header = 0
        
        # Get all huffman tables on the data
        while (data_header < data_size):
            table_destination = data[data_header]
            data_header += 1

            # Count how many codes of each length there are
            codes_count = {
                bit_length: count
                for bit_length, count
                in zip(range(1, 17), data[data_header : data_header+16])
            }
            data_header += 16

            # Get the Huffman values (HUFFVAL)
            huffval_dict = {}   # Dictionary that associates each code bit-length to all its respective Huffman values

            for bit_length, count in codes_count.items():
                huffval_dict.update(
                    {bit_length: data[data_header : data_header+count]}
                )
                data_header += count
            
            # Error checking
            if (data_header > data_size):
                # If we tried to read more bytes than what the data has, then something is wrong with the file
                raise CorruptedJpeg("Failed to parse Huffman tables.")
            
            # Build the Huffman tree
            huffman_tree = {}

            code = 0
            for bit_length, values_list in huffval_dict.items():
                code <<= 1
                for huffval in values_list:
                    code_string = bin(code)[2:].rjust(bit_length, "0")
                    huffman_tree.update({code_string: huffval})
                    code += 1
            
            # Add tree to the Huffman table dictionary
            self.huffman_tables.update({table_destination: huffman_tree})

        # Move the file header to the end of the data segment
        self.file_header += data_size

    def define_quantization_table(self, data:bytes) -> None:
        data_size = len(data)
        data_header = 0

        # Get all quantization tables on the data
        while (data_header < data_size):
            table_destination = data[data_header]
            data_header += 1

            # Get the 64 values of the 8 x 8 quantization table
            qt_values = [value for value in data[data_header : data_header+64]]
            try:
                quantization_table = np.array(qt_values, dtype="int16").reshape(8, 8)
            except ValueError:
                raise CorruptedJpeg("Failed to parse quantization tables.")
            data_header += 64

            # Add the table to the quantization tables dictionary
            self.quantization_tables.update({table_destination: quantization_table})
        
        # Move the file header to the end of the data segment
        self.file_header += data_size

    def define_restart_interval(self, data:bytes) -> None:
        self.restart_interval = bytes_to_uint(data[:2])
        self.file_header += 2

    def start_of_scan(self, data:bytes) -> None:
        data_size = len(data)
        data_header = 0

        # Number of color components in the scan
        components_amount = data[data_header]
        data_header += 1

        # Get parameters of the components in the scan
        my_huffman_tables = {}
        for component in range(components_amount):
            component_id = data[data_header]    # Should match the component ID's on the 'start of frame'
            data_header += 1

            # Selector for the Huffman tables
            tables = data[data_header]
            data_header += 1
            dc_table =  tables >> 4             # Should match the tables ID's on the 'detect huffman table'
            ac_table = (tables & 0x0F) | 0x10
            """NOTE
            The ID of the AC tables is a byte value which the first hexadecimal digit is 1.
            Usually: 0x10 and 0x11 (16 and 17, in decimal).

            On the other hand, the ID of the DC tables begin with hexadecimal digit 0.
            Usually: 0x00 and 0x01.
            """

            # Store the parameters
            my_huffman_tables.update({component_id: HuffmanTable(dc=dc_table, ac=ac_table)})
        
        # Move the file header to the begining of the entropy encoded segment
        self.file_header += data_size

        # Define number of lines
        if self.image_height == 0:
            dnl_index = self.raw_file[self.file_header:].find(DNL)
            if dnl_index != -1:
                dnl_index += self.file_header
                self.image_height = bytes_to_uint(self.raw_file[dnl_index+2 : dnl_index+4])
            else:
                raise CorruptedJpeg("Image height cannot be zero.")

        # Begin the scan of the entropy encoded segment
        if self.scan_mode == "baseline_dct":
            self.baseline_dct_scan(my_huffman_tables)
        elif self.scan_mode == "progressive_dct":
            self.progressive_dct_scan(data)
        else:
            raise UnsupportedJpeg("Encoding mode not supported. Only 'Baseline DCT' and 'Progressive DCT' are supported.")
    
    def bits_generator(self) -> Callable[[int, bool], str]:
        """Returns a function that fetches the bits values in order from the raw file.
        """
        bit_queue = deque()

        # This nested function "remembers" the contents of bit_queue between different calls        
        def get_bits(amount:int=1, restart:bool=False) -> str:
            """Fetches a certain amount of bits from the raw file, and moves the file header
            when a new byte is reached.
            """
            nonlocal bit_queue
            
            # Should be set to 'True' when the restart interval is reached
            if restart:
                bit_queue.clear()       # Discard the remaining bits
                self.file_header += 2   # Jump over the restart marker
            
            # Fetch more bits if the queue has less than the requested amount
            while amount > len(bit_queue):
                next_byte = self.raw_file[self.file_header]
                self.file_header += 1
                
                if next_byte == 0xFF:
                    self.file_header += 1        # Jump over the stuffed byte
                
                bit_queue.extend(
                    np.unpackbits(
                        bytearray((next_byte,))  # Unpack the bits and add them to the end of the queue
                    )
                )
            
            # Return the bits sequence as a string
            return "".join(str(bit_queue.popleft()) for bit in range(amount))
        
        # Return the nested function
        return get_bits

    def baseline_dct_scan(self, huffman_tables_id:dict) -> None:
        """Decode the image data from the entropy encoded segment.

        The file header is should be at the beginning of said segment, and at
        the after the decoding the header will be at the end of the segment.
        """
        # Function to read the bits from the file's bytes
        next_bits = self.bits_generator()

        # Function to decode the next Huffman value
        def next_huffval() -> int:
            codeword = ""
            huffman_value = None

            while huffman_value is None:
                codeword += next_bits()
                if len(codeword) > 16:
                    raise CorruptedJpeg(f"Failed to decode image ({current_mcu}/{mcu_count} MCU's decoded).")
                huffman_value = huffman_table.get(codeword)
            
            return huffman_value

        # Dimensions of the MCU (minimum coding unit)
        mcu_width:int = 8 * max(component.horizontal_sampling for component in self.color_components.values())
        mcu_height:int = 8 * max(component.vertical_sampling for component in self.color_components.values())

        # Amount of MCU's in the whole image
        mcu_count_h = (self.image_width // mcu_width) + (0 if self.image_width % mcu_width == 0 else 1)
        mcu_count_v = (self.image_height // mcu_height) + (0 if self.image_height % mcu_height == 0 else 1)
        mcu_count = mcu_count_h * mcu_count_v

        # 3-dimensional array to store the color values of each pixel on the image
        # array(x-coordinate, y-coordinate, RBG-color)
        array_width = mcu_width * mcu_count_h
        array_height = mcu_height * mcu_count_v
        array_depth = len(self.color_components)
        self.image_array = np.zeros(shape=(array_width, array_height, array_depth), dtype="uint8")

        # Decode all the MCU's in the entropy encoded data
        current_mcu = 0
        previous_dc = 0
        while (current_mcu < mcu_count):
            
            # Current MCU
            mcu = np.zeros(shape=(mcu_width, mcu_height), dtype="int16")
            
            # Loop through all color components
            for depth, (component_id, component) in enumerate(self.color_components.items()):

                # Quantization table of the color component
                quantization_table = self.quantization_tables[component.quantization_table_id]
                
                for _ in range(component.repeat):
                    # Block of 8 x 8 pixels for the color component
                    block = np.zeros(64, dtype="int16")
                    
                    # DC value of the block
                    table_id = huffman_tables_id[component_id].dc
                    huffman_table:dict = self.huffman_tables[table_id]
                    huffman_value = next_huffval()
                    """NOTE
                    For the DC values decoding, the huffman_value represents the bit-length
                    of the next DC value.
                    """
                    
                    dc_value = bin_twos_complement(next_bits(huffman_value)) + previous_dc
                    previous_dc = dc_value
                    block[0] = dc_value

                    # AC values of the block
                    table_id = huffman_tables_id[component_id].ac
                    huffman_table:dict = self.huffman_tables[table_id]
                    index = 1
                    while index < 64:
                        huffman_value = next_huffval()
                        """NOTE
                        The huffman_value is one byte long. For the AC value decoding, the eight bits
                        of the huffman value are in the following format:
                            RRRRSSSS
                        Where:
                            RRRR represents the amount of zeroes before the next non-zero AC value
                            SSSS represents the bit-length of the next AC value
                        
                        A huffman_value of 0x00, however, has a different meaning: it marks the end of
                        the block (all remaining AC values of the block are zero).
                        """
                        
                        # A huffman_value of 0 means the 'end of block' (all remaining AC values are zero)
                        if huffman_value == 0x00:
                            break
                        
                        # Amount of zeroes before the next AC value
                        zero_run_length = huffman_value >> 4
                        index += zero_run_length
                        if index >= 64:
                            break

                        # Get the AC value
                        ac_bit_length = huffman_value & 0x0F
                        
                        if ac_bit_length > 0:
                            ac_value = bin_twos_complement(next_bits(ac_bit_length))
                            block[index] = ac_value
                        
                        # Go to the next AC value
                        index += 1
                    
                    # Undo the zigzag scan and apply dequantization
                    block = undo_zigzag(block) * quantization_table
            
                # Stack and group blocks...
                # (to do)
            
            # Go to the next MCU
            current_mcu += 1
            # print(f"{current_mcu}/{mcu_count}", end="\r")
            
            # Check for restart interval
            if (self.restart_interval > 0) and (current_mcu % self.restart_interval == 0):
                next_bits(amount=0, restart=True)
                previous_dc = 0

    def progressive_dct_scan(self, data:bytes) -> None:
        pass

    def end_of_image(self, data:bytes) -> None:
        self.scan_finished = True
        del self.raw_file


# ----------------------------------------------------------------------------
# Helper functions

def bytes_to_uint(bytes_obj:bytes) -> int:
    return int.from_bytes(bytes_obj, byteorder="big", signed=False)

def bin_twos_complement(bits:str) -> int:
    if bits == "":
        return 0
    elif bits[0] == "1":
        return int(bits, 2)
    elif bits[0] == "0":
        bit_length = len(bits)
        return int(bits, 2) - (2**bit_length - 1)
    else:
        raise ValueError(f"'{bits}' is not a binary number.")

def undo_zigzag(block:np.ndarray) -> np.ndarray:
    return np.array(
        [[block[0], block[1], block[5], block[6], block[14], block[15], block[27], block[28]],
        [block[2], block[4], block[7], block[13], block[16], block[26], block[29], block[42]],
        [block[3], block[8], block[12], block[17], block[25], block[30], block[41], block[43]],
        [block[9], block[11], block[18], block[24], block[31], block[40], block[44], block[53]],
        [block[10], block[19], block[23], block[32], block[39], block[45], block[52], block[54]],
        [block[20], block[22], block[33], block[38], block[46], block[51], block[55], block[60]],
        [block[21], block[34], block[37], block[47], block[50], block[56], block[59], block[61]],
        [block[35], block[36], block[48], block[49], block[57], block[58], block[62], block[63]]],
        dtype=block.dtype
    )


# ----------------------------------------------------------------------------
# Decoder exceptions

class JpegError(Exception):
    """Parent of all other exceptions of this decoder."""


class NotJpeg(JpegError):
    """File is not a JPEG image."""

class CorruptedJpeg(JpegError):
    """Failed to parse the file headers."""

class UnsupportedJpeg(JpegError):
    """JPEG image is encoded in a way that our decoder does not support."""


# ----------------------------------------------------------------------------
# Run script

if __name__ == "__main__":
    # jpeg = JpegDecoder(r"C:\Users\Tiago\OneDrive\Documentos\Python\Projetos\Steganography\Tiago (2).jpg")
    jpeg = JpegDecoder(r"C:\Users\Tiago\Pictures\ecce_homo_antonio_ciseri_1880.jpg")
    