import numpy as np
from collections import deque, namedtuple
from itertools import product
from math import ceil, cos, pi
from scipy.interpolate import griddata
from typing import Callable, Tuple, Union
from pathlib import Path

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
ColorComponent = namedtuple("ColorComponent", "name order vertical_sampling horizontal_sampling quantization_table_id repeat shape")
HuffmanTable = namedtuple("HuffmanTable", "dc ac")

class JpegDecoder():

    def __init__(self, file:Path) -> None:

        # Open file
        with open(file, "rb") as image:
            self.raw_file = image.read()
        self.file_size = len(self.raw_file)     # Size (in bytes) of the file
        self.file_path = file if isinstance(file, Path) else Path(file)
        
        # Check if file is a JPEG image
        # (The file needs to start with bytes 'FFD8FF')
        if not self.raw_file.startswith(SOI + b"\xFF"):
            raise NotJpeg("File is not a JPEG image.")
        print(f"Reading file '{file.name}' ({self.file_size:,} bytes)")

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
        self.sample_shape = ()          # Size to upsample the subsampled color components
        self.huffman_tables = {}        # Hold all huffman tables
        self.quantization_tables = {}   # Hold all quantization tables
        self.restart_interval = 0       # How many MCUs before each restart marker
        self.image_array = None         # Store the color values for each pixel of the image
        self.scan_count = 0             # Counter for the performed scans

        # Main loop to find and process the supported file segments
        """NOTE
        We are sequentially looking for markers on the file. Once a recognized
        marker is found, control is passed to a method to handle it. The method
        then gives the control back to the main loop, which continues from where
        the method stopped.
        If the marker isn't recognized, then its data segment is just skipped.
        """
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
            print("Scan mode: Sequential")
        elif mode == SOF2:
            self.scan_mode = "progressive_dct"
            print("Scan mode: Progressive")
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
        If the height is specified as zero here, then it means that the height value
        is going to be specidied on the DNL segment after the first scan.
        This is for the case when the final height is unknown when the image is
        being created, for example when a scanner is generating the image.
        """

        self.image_width = bytes_to_uint(data[data_header : data_header+2])
        data_header += 2
        print(f"Image dimensions: {self.image_width} x {self.image_height}")

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

        if components_amount == 3:
            print("Color space: YCbCr")
        else:
            print("Color space: greyscale")

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
                    name = component,                                    # Name of the color component
                    order = count-1,                                        # Order in which the component will come in the image
                    horizontal_sampling = horizontal_sample,                # Amount of pixels sampled in the horizontal
                    vertical_sampling = vertical_sample,                    # Amount of pixels sampled in the vertical
                    quantization_table_id = my_quantization_table,          # Quantization table selector
                    repeat = horizontal_sample * vertical_sample,           # Amount of times the component repeats during decoding
                    shape = (8*horizontal_sample, 8*vertical_sample),       # Dimensions (in pixels) of the MCU for the component
                )

                # Add the component parameters to the dictionary
                self.color_components.update({my_id: my_component})

                # Have we parsed all components?
                if count == components_amount:
                    break
        
        except IndexError:
            raise CorruptedJpeg("Failed to parse the start of frame.")
        
        # Shape of the sampling area
        # (these values will be used to upsample the subsampled color components)
        sample_width = max(component.shape[0] for component in self.color_components.values())
        sample_height = max(component.shape[1] for component in self.color_components.values())
        self.sample_shape = (sample_width, sample_height)

        # Display the samplings
        print(f"Horizontal sampling: {' x '.join(str(component.horizontal_sampling) for component in self.color_components.values())}")
        print(f"Vertical sampling  : {' x '.join(str(component.vertical_sampling) for component in self.color_components.values())}")
        
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
            print(f"Parsed Huffman table - ", end="")
            print(f"ID: {table_destination & 0x0F} ({'DC' if table_destination >> 4 == 0 else 'AC'})")

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
            qt_values = np.array([value for value in data[data_header : data_header+64]], dtype="int16")
            try:
                quantization_table = undo_zigzag(qt_values)
            except ValueError:
                raise CorruptedJpeg("Failed to parse quantization tables.")
            data_header += 64

            # Add the table to the quantization tables dictionary
            self.quantization_tables.update({table_destination: quantization_table})
            print(f"Parsed quantization table - ID: {table_destination}")
        
        # Move the file header to the end of the data segment
        self.file_header += data_size

    def define_restart_interval(self, data:bytes) -> None:
        self.restart_interval = bytes_to_uint(data[:2])
        self.file_header += 2
        print(f"Restart interval: {self.restart_interval}")

    def start_of_scan(self, data:bytes) -> None:
        data_size = len(data)
        data_header = 0

        # Number of color components in the scan
        components_amount = data[data_header]
        data_header += 1

        # Get parameters of the components in the scan
        my_huffman_tables = {}
        my_color_components = {}
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
            my_color_components.update({component_id: self.color_components[component_id]})
        
        # Get spectral selection and successive approximation
        if self.scan_mode == "progressive_dct":
            spectral_selection_start = data[data_header]    # Index of the first values of the data unit
            spectral_selection_end = data[data_header+1]    # Index of the last values of the data unit
            bit_position_high = data[data_header+2] >> 4    # The position of the last bit sent in the previous scan
            bit_position_low = data[data_header+2] & 0x0F   # The position of the bit sent in the current scan
            """NOTE
            The data unit is formed by blocks of 8 x 8 pixels (64 values in total, indexed from 0 to 63).
            The progressive scan breaks the values in different scans. And the bits of those values can
            also be broken in separate scans. It is up to the encoder to decide how to split the data.
            After all scans, it should be possible to reconstruct all values of all data units.
            """
            data_header += 3
        
        # Move the file header to the begining of the entropy encoded segment
        self.file_header += data_size

        # Define number of lines
        if self.image_height == 0:
            dnl_index = self.raw_file[self.file_header:].find(DNL)
            if dnl_index != -1:
                dnl_index += self.file_header
                self.image_height = bytes_to_uint(self.raw_file[dnl_index+4 : dnl_index+6])
            else:
                raise CorruptedJpeg("Image height cannot be zero.")

        # Dimensions of the MCU (minimum coding unit)
        if components_amount > 1:
            self.mcu_width:int = 8 * max(component.horizontal_sampling for component in self.color_components.values())
            self.mcu_height:int = 8 * max(component.vertical_sampling for component in self.color_components.values())
            self.mcu_shape = (self.mcu_width, self.mcu_height)
        else:
            self.mcu_width:int = 8
            self.mcu_height:int = 8
            self.mcu_shape = (8, 8)
        """NOTE
        If there is only one color component in the scan, then the MCU size is always 8 x 8.

        If there is more than one color component, the MCU size is determined by the component
        with the highest resolution (considering all the components of the image, not only the
        components in the scan).
        """

        # Amount of MCUs in the whole image (horizontal, vertical, and total)
        if components_amount > 1:
            self.mcu_count_h = (self.image_width // self.mcu_width) + (0 if self.image_width % self.mcu_width == 0 else 1)
            self.mcu_count_v = (self.image_height // self.mcu_height) + (0 if self.image_height % self.mcu_height == 0 else 1)
        else:
            component = my_color_components[component_id]
            sample_ratio_h = self.sample_shape[0] / component.shape[0]
            sample_ratio_v = self.sample_shape[1] / component.shape[1]
            layer_width = self.image_width / sample_ratio_h
            layer_height = self.image_height / sample_ratio_v
            self.mcu_count_h = ceil(layer_width / self.mcu_width)
            self.mcu_count_v = ceil(layer_height / self.mcu_height)
        
        self.mcu_count = self.mcu_count_h * self.mcu_count_v

        # Create the image array (if one does not exist already)
        if self.image_array is None:
            # 3-dimensional array to store the color values of each pixel on the image
            # array(x-coordinate, y-coordinate, RBG-color)
            count_h = (self.image_width // self.sample_shape[0]) + (0 if self.image_width % self.sample_shape[0] == 0 else 1)
            count_v = (self.image_height // self.sample_shape[1]) + (0 if self.image_height % self.sample_shape[1] == 0 else 1)
            self.array_width = self.sample_shape[0] * count_h
            self.array_height = self.sample_shape[1] * count_v
            self.array_depth = len(self.color_components)
            self.image_array = np.zeros(shape=(self.array_width, self.array_height, self.array_depth), dtype="int16")
        
        # Setup scan counter
        if self.scan_count == 0:
            self.scan_amount = self.raw_file[self.file_header:].count(SOS) + 1
            print(f"Number of scans: {self.scan_amount}")

        # Begin the scan of the entropy encoded segment
        if self.scan_mode == "baseline_dct":
            self.baseline_dct_scan(my_huffman_tables, my_color_components)
        elif self.scan_mode == "progressive_dct":
            self.progressive_dct_scan(
                my_huffman_tables,
                my_color_components,
                spectral_selection_start,
                spectral_selection_end,
                bit_position_high,
                bit_position_low
            )
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

    def baseline_dct_scan(self, huffman_tables_id:dict, my_color_components:dict) -> None:
        """Decode the image data from the entropy encoded segment.

        The file header should be at the beginning of said segment, and at
        the after the decoding the header will be moved to the end of the segment.
        """
        print(f"\nScan {self.scan_count+1} of {self.scan_amount}")
        print(f"Color components: {', '.join(component.name for component in my_color_components.values())}")
        print(f"MCU count: {self.mcu_count}")
        print(f"Decoding MCUs and performing IDCT...")

        # Function to read the bits from the file's bytes
        next_bits = self.bits_generator()

        # Function to decode the next Huffman value
        def next_huffval() -> int:
            codeword = ""
            huffman_value = None

            while huffman_value is None:
                codeword += next_bits()
                if len(codeword) > 16:
                    raise CorruptedJpeg(f"Failed to decode image ({current_mcu}/{self.mcu_count} MCUs decoded).")
                huffman_value = huffman_table.get(codeword)
            
            return huffman_value

        # Function to perform the inverse discrete cosine transform (IDCT)
        idct = InverseDCT()

        # Function to resize a block of color values
        resize = ResizeGrid()

        # Number of color components in the scan
        components_amount = len(my_color_components)
        
        # Decode all MCUs in the entropy encoded data
        current_mcu = 0
        previous_dc = np.zeros(components_amount, dtype="int16")
        while (current_mcu < self.mcu_count):
            
            # (x, y) coordinates, on the image, for the current MCU
            mcu_y, mcu_x = divmod(current_mcu, self.mcu_count_h)
            
            # Loop through all color components
            for depth, (component_id, component) in enumerate(my_color_components.items()):

                # Quantization table of the color component
                quantization_table = self.quantization_tables[component.quantization_table_id]

                # Minimum coding unit (MCU) of the component
                if components_amount > 1:
                    my_mcu = np.zeros(shape=component.shape, dtype="int16")
                    repeat = component.repeat
                else:
                    my_mcu = np.zeros(shape=(8, 8), dtype="int16")
                    repeat = 1
                """NOTE
                When there is more than one color component in the scan, the components are
                interleaved. And the component with the highest resolution is repeated enough
                times to cover the area of the subsampled components.
                """
                
                for block_count in range(repeat):
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
                    
                    dc_value = bin_twos_complement(next_bits(huffman_value)) + previous_dc[depth]
                    previous_dc[depth] = dc_value
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

                    # Apply the inverse discrete cosine transform (IDCT)
                    block = idct(block)

                    # Coordinates of the block on the current MCU
                    block_y, block_x = divmod(block_count, component.horizontal_sampling)
                    block_y, block_x = 8*block_y, 8*block_x

                    # Add the block to the MCU
                    my_mcu[block_x : block_x+8, block_y : block_y+8] = block
            
                # Upsample the block if necessary
                if component.shape != self.sample_shape:
                    my_mcu = resize(my_mcu, self.sample_shape)
                """NOTE
                Linear interpolation is performed on subsampled color components.
                """
                
                # Add the MCU to the image
                x = self.mcu_width * mcu_x
                y = self.mcu_height * mcu_y
                self.image_array[x : x+self.mcu_width, y : y+self.mcu_height, component.order] = my_mcu
            
            # Go to the next MCU
            current_mcu += 1
            print_progress(current_mcu, self.mcu_count)
            
            # Check for restart interval
            if (self.restart_interval > 0) and (current_mcu % self.restart_interval == 0) and (current_mcu != self.mcu_count):
                next_bits(amount=0, restart=True)
                previous_dc[:] = 0
            """NOTE
            When the Restart Interval is reached, the previous DC values are reseted to zero
            and the file header is moved to the byte boundary after the marker.
            """
        self.scan_count += 1
        print_progress(current_mcu, self.mcu_count, done=True)

    def progressive_dct_scan(self,
        huffman_tables_id:dict,
        my_color_components:dict,
        spectral_selection_start:int,
        spectral_selection_end:int,
        bit_position_high:int,
        bit_position_low:int) -> None:

        # Whether to the scan contains DC or AC values
        if (spectral_selection_start == 0) and (spectral_selection_end == 0):
            values = "dc"
        elif (spectral_selection_start > 0) and (spectral_selection_end >= spectral_selection_start):
            values = "ac"
        else:
            raise CorruptedJpeg("Progressive JPEG images cannot contain both DC and AC values in the same scan.")
        """NOTE
        In sequential JPEG both DC and AC values come in the same scan, however in progressive JPEG
        they must come in different scans.
        """
        
        # Whether this is a refining scan
        if bit_position_high == 0:
            refining = False
        elif (bit_position_high - bit_position_low) == 1:
            refining = True
        else:
            raise CorruptedJpeg("Progressive JPEG images cannot contain more than 1 bit for each value on a refining scan.")
        """NOTE
        The first scan of a value sends a certain amount of the value's most significant bits.
        The following scans of the same value send the next bits, in order, one bit per scan.
        Those scans are called "refining scans".
        """
        print(f"\nScan {self.scan_count+1} of {self.scan_amount}")
        print(f"Color components: {', '.join(component.name for component in my_color_components.values())}")
        print(f"Spectral selection: {spectral_selection_start}-{spectral_selection_end} ({values.upper()})")
        print(f"Successive approximation: {bit_position_high}-{bit_position_low} ({'refining' if refining else 'first'} scan)")
        print(f"MCU count: {self.mcu_count}")
        print(f"Decoding MCUs...")

        # Function to read the bits from the file's bytes
        next_bits = self.bits_generator()

        # Function to decode the next Huffman value
        def next_huffval() -> int:
            codeword = ""
            huffman_value = None

            while huffman_value is None:
                codeword += next_bits()
                if len(codeword) > 16:
                    raise CorruptedJpeg(f"Failed to decode image ({current_mcu}/{self.mcu_count} MCUs decoded).")
                huffman_value = huffman_table.get(codeword)
            
            return huffman_value

        # Beginning of scan
        current_mcu = 0
        components_amount = len(my_color_components)
        if (values == "ac") and (components_amount > 1):
            raise CorruptedJpeg("An AC progressive scan can only have a single color component.")
        """NOTE
        A DC progressive scan can have more than one color component, while an AC progressive
        scan must have only one color component.
        """

        # DC values scan
        if values == "dc":
            
            # First scan (DC)
            """NOTE
            For the most part, the first DC scan on progressive mode is the same as on baseline mode.
            The only difference is that the decoded value needs to have to undergo through a left
            bit shift by the amount specified in 'bit_position_low', because the progressive scan
            only gives this amount of the first bits of the value on the first scan.
            """
            if not refining:
                # Previous DC values
                previous_dc = np.zeros(components_amount, dtype="int16")

            while (current_mcu < self.mcu_count):
                
                # Loop through all color components
                for depth, (component_id, component) in enumerate(my_color_components.items()):

                    # (x, y) coordinates, on the image, for the current MCU
                    x = (current_mcu % self.mcu_count_h) * component.shape[0]
                    y = (current_mcu // self.mcu_count_h) * component.shape[1]

                    # Minimum coding unit (MCU) of the component
                    if components_amount > 1:
                        repeat = component.repeat
                    else:
                        repeat = 1
                    
                    # Blocks of 8 x 8 pixels for the color component
                    for block_count in range(repeat):

                        # Coordinates of the block on the current MCU
                        block_y, block_x = divmod(block_count, component.horizontal_sampling)
                        delta_y, delta_x = 8*block_y, 8*block_x
                        
                        # First scan of the DC values
                        if not refining:
                            # DC value of the block
                            table_id = huffman_tables_id[component_id].dc
                            huffman_table:dict = self.huffman_tables[table_id]
                            huffman_value = next_huffval()
                            
                            # Get the DC value (partial)
                            dc_value = bin_twos_complement(next_bits(huffman_value)) + previous_dc[depth]
                            previous_dc[depth] = dc_value
                            
                            # Store the partial DC value on the image array
                            self.image_array[x+delta_x, y+delta_y, component.order] = (dc_value << bit_position_low)
                            """NOTE
                            'bit_position_low' is the position of the last value's bit sent in the scan.
                            So the partial value has its bits left-shifted by this amount.
                            """
                        
                        # Refining scan for the DC values
                        else:
                            new_bit = int(next_bits())
                            self.image_array[x+delta_x, y+delta_y, component.order] |= (new_bit << bit_position_low)
                            """NOTE
                            A refining scan of the DC values just sent the next bit of each value, in the
                            same order as the partial values were sent.
                            So the bit is just OR'ed to the existing values, in the position the bit belongs.
                            """
                
                # Go to the next MCU
                current_mcu += 1
                print_progress(current_mcu, self.mcu_count)
                
                # Check for restart interval
                if (self.restart_interval > 0) and (current_mcu % self.restart_interval == 0) and (current_mcu != self.mcu_count):
                    next_bits(amount=0, restart=True)
                    if not refining:
                        previous_dc[:] = 0

        # AC values scan
        elif values == "ac":
            """NOTE
            This scan always has one color component, and the MCU always is 8x8 pixels.
            All the AC values are considered to be in one contiguous band.
            
            The band starts with the values specified on the spectral selection for the first MCU,
            then those values for the next MCU, and so on until the whole image is covered.
            
            The order of the MCUs is: left-to-right starting from the top left, then top-to-bottom.
            """
            # Spectral selection
            spectral_size = (spectral_selection_end + 1) - spectral_selection_start
            
            # Color component
            (component_id, component), = my_color_components.items()

            # Huffman table
            table_id = huffman_tables_id[component_id].ac
            huffman_table:dict = self.huffman_tables[table_id]

            # End of band run length
            eob_run = 0
            """NOTE
            It is the amount of bands that the decoder needs to skip during decoding.
            (a band is the section of a MCU, as specified in the spectral selection)
            On the first scan, all values in those bands are considered to be zero.
            On refining scans, the non-zero values that were skiped will be refined.
            """

            # Zero run length
            zero_run = 0
            """NOTE
            This is the amount of zero values to be skipped. If the run goes beyond
            one band, then the run is finished.
            
            On refining scans, the non-zero values found along the way will be
            refined (those values do not decrease the zero_run counter).
            """

            # Refining function
            def refine_ac() -> None:
                """Perform the refinement of the AC values on a progressive scan
                """
                nonlocal to_refine, next_bits, bit_position_low, component
                
                # Fetch the bits that will be used to refine the AC values
                # (the bits come in the same order that the values to be refined were found)
                refine_bits = next_bits(len(to_refine))

                # Refine the AC values
                ref_index = 0
                while to_refine:
                    ref_x, ref_y = to_refine.popleft()
                    new_bit = int(refine_bits[ref_index], 2)
                    self.image_array[ref_x, ref_y, component.order] |= new_bit << bit_position_low
                    ref_index += 1

            # Queue of AC values that will be refined
            to_refine = deque()

            # Decode and refine the AC values
            current_mcu = 0
            while (current_mcu < self.mcu_count):

                # Coordinates of the MCU on the image
                x = (current_mcu % self.mcu_count_h) * 8
                y = (current_mcu // self.mcu_count_h) * 8

                # Loop through the band
                index = spectral_selection_start
                while index <= spectral_selection_end:  # The element at the end of the band is included
                    
                    # Get the next Huffman value from the encoded data
                    huffman_value = next_huffval()
                    run_magnitute = huffman_value >> 4
                    ac_bits_length = huffman_value & 0x0F
                    
                    # Determine the run length
                    if huffman_value == 0:
                        # End of band run of 1
                        eob_run = 1
                        break
                    elif huffman_value == 0xF0:
                        zero_run = 16
                    elif (ac_bits_length == 0):
                        # End of band run (length determined by the next bits on the data)
                        # (amount of bands to skip)
                        eob_bits = next_bits(run_magnitute)
                        eob_run = (1 << run_magnitute) + int(eob_bits, 2)
                        break
                        """NOTE (EOB run)
                        If the upper lower of the Huffman value is 0x0, and the upper nibble is from 0x0 to 0xE,
                        then a End of Band Run (EOB run) is defined. The EOB run tells the decoder how many
                        bands to skip.

                        This amount is determined by the Huffman value and the bits following it. The upper nibble
                        of the Huffman value determines the amplitude (N) of the EOB run (2^N). Then the next N bits
                        on the data (following the Huffman value) determine the length to be added to the EOB run.
                        Those bits are converted from binary to decimal and added to 2^N:
                        EOB run = 2^N + length
                        """
                    else:
                        # Amount of zero values to skip
                        zero_run = run_magnitute
                        """NOTE (Zero run)
                        If the lower nibble of the Huffman value greater than 0x0 (except for 0xF0), then a zero run
                        is defined. The zero run is the amount of zero values to skip within a band. This amount is
                        determined directly by the value of the upper nibble of the Huffman value.
                        
                        The lower nibble determines the bit-length (N) of the next non-zero AC value.
                        The next N bits on the data (following the Huffman value) are the next AC value,
                        in a binary two's complement representation.

                        A Huffman value of 0xF0 defines a zero run of length 16, with no AC value bits following it.
                        """
                    
                    # Perform the zero run
                    if not refining and zero_run:   # First scan
                        index += zero_run
                        zero_run = 0
                        """NOTE
                        On the first scan, all AC values skipped are considered to be zero.
                        """
                    else:
                        while zero_run > 0:         # Refining scan
                            xr, yr = zagzig[index]
                            current_value = self.image_array[x + xr, y + yr, component.order]
                            
                            if current_value == 0:
                                zero_run -= 1
                            else:
                                to_refine.append((x + xr, y + yr))
                            
                            index += 1
                            """NOTE
                            On a refining scan, only the zero AC values decrease the zero run counter.
                            The decoder keeps moving to the next AC value until the counter is depleted.
                            The non-zero values found along the way are enqueued to be refined.
                            """
                    
                    # Decode the next AC value
                    if ac_bits_length > 0:
                        ac_bits = next_bits(ac_bits_length)
                        ac_value = bin_twos_complement(ac_bits)
                        
                        # Store the AC value on the image array
                        # (the zig-zag scan order is undone to find the position of the value on the image)
                        ac_x, ac_y = zagzig[index]

                        # In order to create a new AC value, the decoder needs to be at a zero value
                        # (the index is moved until a zero is found, other values along the way will be refined)
                        if refining:
                            while self.image_array[x + ac_x, y + ac_y, component.order] != 0:
                                to_refine.append((x + ac_x, y + ac_y))
                                index += 1
                                ac_x, ac_y = zagzig[index]
                                """NOTE
                                On a refining scan, when the lower nibble of the Huffman value is not zero
                                then a new AC value is created. However the new AC value cannot be created
                                in the spot where there is an existing AC value. So if the decoder happens
                                to be at a non-zero AC value, then it moves to the next spot until a zero
                                is found. The non-zero values found along the way are enqueued to be refined.
                                """
                        
                        # Create a new ac_value
                        self.image_array[x + ac_x, y + ac_y, component.order] = ac_value << bit_position_low
                        
                        # Move to the next value
                        index += 1
                    
                    # Refine AC values skipped by the zero run
                    if refining:
                        refine_ac()
                        """NOTE
                        Following the bits of the AC value (on the image data), come the bits of all
                        values enqueued to be refined. One bit per value, in the same order the values
                        were found. So if N values are going to be refined, then N bits will follow.
                        """
                
                # Move to the next band if we are at the end of a band
                if index > spectral_selection_end:
                    current_mcu += 1
                    if refining:
                        # Coordinates of the MCU on the image
                        x = (current_mcu % self.mcu_count_h) * 8
                        y = (current_mcu // self.mcu_count_h) * 8

                # Perform the end of band run
                if not refining:            # First scan
                    current_mcu += eob_run
                    eob_run = 0
                    """NOTE
                    In the first scan, all the skipped AC values are consideded to be zero.
                    If the EOB run is called when a band has been partially processed, then
                    only the remaining values on the band are considered zero (this band
                    stills count for the EOB run counter).
                    """
                
                else:                       # Refining scan
                    while eob_run > 0:
                        xr, yr = zagzig[index]
                        current_value = self.image_array[x + xr, y + yr, component.order]
                        
                        if current_value != 0:
                            to_refine.append((x + xr, y + yr))
                        
                        index += 1
                        if index > spectral_selection_end:
                            
                            # Move to the next band
                            eob_run -= 1
                            current_mcu += 1
                            index = spectral_selection_start
                            
                            # Coordinates of the MCU on the image
                            x = (current_mcu % self.mcu_count_h) * 8
                            y = (current_mcu // self.mcu_count_h) * 8
                    """NOTE
                    On a refining scan, the non-zero values that were skipped are enqueued
                    to be refined. If the EOB run begins on a partially processed band, then
                    only the remaining values on the band are considered (this band still
                    decreases the EOB run counter).
                    """
                
                # Refine the AC values found during the EOB run
                if refining:
                    refine_ac()
                    """NOTE
                    On the image data stream, the bits following the Huffman value that defined
                    an EOB run will be used to refine the non-zero AC values that were skipped
                    during the run. If N non-zero values were skipped, then N bits will follow
                    to refine them. One bit per value, in the same order the values were found.
                    """
                
                print_progress(current_mcu, self.mcu_count)

                 # Check for restart interval
                if (self.restart_interval > 0) and (current_mcu % self.restart_interval == 0) and (current_mcu != self.mcu_count):
                    next_bits(amount=0, restart=True)
                    """NOTE
                    When a restart interval is reashed, then the decoder moves to the next byte
                    boundary (if not already at one), and then jumps over the restart marker (2 bytes long).
                    """
        
        print_progress(current_mcu, self.mcu_count, done=True)
        
        # Check if all scans have been performed and perform the IDCT
        self.scan_count += 1
        if self.scan_count == self.scan_amount:

            # Function to perform the inverse discrete cosine transform (IDCT)
            idct = InverseDCT()

            # Function to resize a block of color values
            resize = ResizeGrid()
            
            # Perform the IDCT once all scans have finished
            dct_array = self.image_array.copy()
            print("\nPerforming IDCT on each color component...")
            for component in self.color_components.values():
                quantization_table = self.quantization_tables[component.quantization_table_id]

                ratio_h = self.sample_shape[0] // component.shape[0]
                ratio_v = self.sample_shape[1] // component.shape[1]
                component_width = self.array_width // ratio_h
                component_height = self.array_height // ratio_v

                mcu_count_h = component_width // 8
                mcu_count_v = component_height // 8
                mcu_count = mcu_count_h * mcu_count_v

                # Perform the inverse discrete cosine transform (IDCT)
                for current_mcu in range(mcu_count):
                    
                    # Get coordinates of the block
                    x1 = (current_mcu % mcu_count_h) * 8
                    y1 = (current_mcu // mcu_count_h) * 8
                    x2 = x1 + 8
                    y2 = y1 + 8

                    # Undo quantization on the block
                    block = dct_array[x1 : x2, y1 : y2, component.order]
                    block *= quantization_table

                    # Perform IDCT on the block to get the color values
                    block = idct(block.reshape(8, 8))

                    # Upsample the block if necessary
                    if component.shape != self.sample_shape:
                        block = resize(block, self.sample_shape)
                        x1 *= ratio_h
                        y1 *= ratio_v
                        x2 *= ratio_h
                        y2 *= ratio_v
                    
                    self.image_array[x1 : x2, y1 : y2, component.order] = block

                    print_progress(current_mcu+1, mcu_count, header=component.name.ljust(2))
                
                print_progress(current_mcu+1, mcu_count, header=component.name.ljust(2), done=True)

    def end_of_image(self, data:bytes) -> None:
        
        # Clip the image array to the image dimensions
        self.image_array = self.image_array[0 : self.image_width, 0 : self.image_height, :]
        """NOTE
        When the image dimensions are not multiples of the MCU size, padding pixels are
        added to the right and bottom edges of the image in order to make the dimensions
        multiples.
        Then it is the duty of the decoder to remove those extra pixels.
        """
        
        # Convert image from YCbCr to RGB
        if (self.array_depth == 3):
            self.image_array = YCbCr_to_RGB(self.image_array)
        elif (self.array_depth == 1):
            np.clip(self.image_array, a_min=0, a_max=255, out=self.image_array)
            self.image_array = self.image_array[..., 0].astype("uint8")
        
        self.scan_finished = True
        self.show()
        del self.raw_file

    def show(self):
        """Display the decoded image in a window.
        """
        try:
            import tkinter as tk
            from tkinter import ttk
        except ModuleNotFoundError:
            self.show2()
            return
        try:
            from PIL import Image
            from PIL.ImageTk import PhotoImage
        except ModuleNotFoundError:
            print("The Pillow module needs to be installed in order to display the rendered image.")
            print("For installing: https://pillow.readthedocs.io/en/stable/installation.html")

        print("\nRendering the decoded image...")

        # Create the window
        window = tk.Tk()
        window.title(f"Decoded JPEG: {self.file_path.name}")
        try:
            window.state("zoomed")
        except tk.TclError:
            window.state("normal")

        # Horizontal and vertical scrollbars
        scrollbar_h = ttk.Scrollbar(orient = tk.HORIZONTAL)
        scrollbar_v = ttk.Scrollbar(orient = tk.VERTICAL)
        
        # Canvas where the image will be drawn
        canvas = tk.Canvas(
            width = self.image_width,
            height = self.image_height,
            scrollregion = (0, 0, self.image_width, self.image_height),
            xscrollcommand = scrollbar_h.set,
            yscrollcommand = scrollbar_v.set,
        )
        scrollbar_h["command"] = canvas.xview
        scrollbar_v["command"] = canvas.yview

        # Button for saving the image
        save_button = ttk.Button(
            command = self.save,
            text = "Save decoded image",
            padding = 1,
        )
        
        # Convert the image array to a format that Tkinter understands
        my_image = PhotoImage(
            Image.fromarray(
                np.swapaxes(self.image_array, 0, 1)
            )
        )

        # Draw the image to the canvas
        canvas.create_image(0, 0, image=my_image, anchor="nw")

        # Add the canvas and scrollbars to the window
        canvas.pack()
        scrollbar_h.pack(
            side = tk.BOTTOM,
            fill = tk.X,
            before = canvas,
        )
        scrollbar_v.pack(
            side = tk.RIGHT,
            fill = tk.Y,
            before = canvas,
        )

        # Add the save button to the window
        save_button.pack(
            side = tk.TOP,
            before = canvas,
        )

        # Open the window
        window.mainloop()

    def show2(self):
        """Display the decoded image in the default image viewer of the operating system.
        """
        try:
            from PIL import Image
        except ModuleNotFoundError:
            print("The Pillow module needs to be installed in order to display the rendered image.")
            print("For installing it: https://pillow.readthedocs.io/en/stable/installation.html")
            return
        
        img = np.swapaxes(self.image_array, 0, 1)
        Image.fromarray(img).show()
    
    def save(self) -> None:
        """Open a file dialog to save the image array as an image to the disk.
        """
        from PIL import Image
        from tkinter.filedialog import asksaveasfilename
        
        # Open a file dialog for the user to provide a path
        img_path = Path(
            asksaveasfilename(
                defaultextension = "png",
                title = "Save decoded image as...",
                filetypes = (
                    ("PNG image", "*.png"),
                    ("Bitmap image", "*.bmp"),
                    ("All files", "*.*")
                ),
                initialfile = self.file_path.stem,
                initialdir = self.file_path.parent,
            )
        )

        # If the user has canceled, then exit the function
        if img_path == Path():
            return
        
        # Make sure that the saved image does not overwrite an existing file
        count = 1
        my_stem = img_path.stem
        while img_path.exists():
            img_path = img_path.with_stem(f"{my_stem} ({count})")
            count += 1
        
        # Convert the image array to a PIL object
        my_image = Image.fromarray(np.swapaxes(self.image_array, 0, 1))

        # Save the image to disk
        try:
            my_image.save(img_path)
        except ValueError:
            img_path = img_path.with_suffix(".png")
            count = 1
            my_stem = img_path.stem
            while img_path.exists():
                img_path = img_path.with_stem(f"{my_stem} ({count})")
                count += 1
            my_image.save(img_path, format="png")
        
        print(f"Decoded image was saved to '{img_path}'")


class InverseDCT():
    # Precalculate the constant values used on the IDCT function
    idct_table = np.zeros(shape=(8,8,8,8), dtype="float64")
    xyuv_coordinates = tuple(product(range(8), repeat=4))
    xy_coordinates = tuple(product(range(8), repeat=2))
    for x, y, u, v in xyuv_coordinates:
        # Scaling factors
        Cu = 2**(-0.5) if u == 0 else 1.0   # Horizontal
        Cv = 2**(-0.5) if v == 0 else 1.0   # Vertical 

        # Frequency component
        idct_table[x, y, u, v] = 0.25 * Cu * Cv * cos((2*x + 1) * pi * u / 16) * cos((2*y + 1) * pi * v / 16)

    def __call__(self, block:np.ndarray) -> np.ndarray:
        """Takes a 8 x 8 array of DCT coefficients, and performs the inverse discrete
        cosine transform in order to reconstruct the color values.
        """
        # Array to store the results
        output = np.zeros(shape=(8, 8), dtype="float64")

        # Summation of the frequecies components
        for x, y in self.xy_coordinates:
            output[x, y] = np.sum(block * self.idct_table[x, y, ...], dtype="float64")
        
        # Return the color values
        return np.round(output).astype(block.dtype) + 128
        """NOTE
        128 is added to the result because, before the foward DCT transform, 128
        was subtracted from the value (in order to center around zero values
        from 0 to 255).
        """

class ResizeGrid():
    mesh_cache = {}
    indices_cache = {}

    def __call__(self, block:np.ndarray, new_shape:Tuple[int,int]) -> np.ndarray:
        """Takes a 2-dimensional array and resizes it while performing
        linear interpolation between the points.
        """
        old_width, old_height = block.shape
        new_width, new_height = new_shape
        key = ((old_width, old_height), (new_width, new_height))

        new_xy = self.mesh_cache.get(key)
        if new_xy is None:
            max_x = old_width - 1
            max_y = old_height - 1
            num_points_x = new_width * 1j
            num_points_y = new_height * 1j
            new_x, new_y = np.mgrid[0 : max_x : num_points_x, 0 : max_y : num_points_y]
            new_xy = (new_x, new_y)
            self.mesh_cache.update({key: new_xy})
        
        old_xy = self.indices_cache.get(key[0])
        if old_xy is None:
            xx, yy = np.indices(block.shape)
            xx, yy = xx.flatten(), yy.flatten()
            old_xy = (xx, yy)
            self.indices_cache.update({key[0]: old_xy})
        
        resized_block = griddata(old_xy, block.ravel(), new_xy)
        
        return np.round(resized_block).astype(block.dtype)


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
    """Takes an 1D array of 64 elements and undo the zig-zag scan of the JPEG
    encoding process. Returns a 2D array (8 x 8) that represents a block of pixels.
    """
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
    ).T # <-- transposes the array
    """NOTE
    The array is transposed so the code above matches the (x, y) positions of the elements
    in the 8 x 8 block of pixels:
    array[x, y] = value on that pixel position
    """

# List that undoes the zig-zag ordering for a single element in a band
# (the element index is used on the list, and it returns a (x, y) tuple
# for the coordinates on the data unit)
zagzig = (
    (0, 0), (1, 0), (0, 1), (0, 2), (1, 1), (2, 0), (3, 0), (2, 1),
    (1, 2), (0, 3), (0, 4), (1, 3), (2, 2), (3, 1), (4, 0), (5, 0),
    (4, 1), (3, 2), (2, 3), (1, 4), (0, 5), (0, 6), (1, 5), (2, 4),
    (3, 3), (4, 2), (5, 1), (6, 0), (7, 0), (6, 1), (5, 2), (4, 3),
    (3, 4), (2, 5), (1, 6), (0, 7), (1, 7), (2, 6), (3, 5), (4, 4),
    (5, 3), (6, 2), (7, 1), (7, 2), (6, 3), (5, 4), (4, 5), (3, 6),
    (2, 7), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (7, 4), (6, 5),
    (5, 6), (4, 7), (5, 7), (6, 6), (7, 5), (7, 6), (6, 7), (7, 7)
)

def YCbCr_to_RGB(image_array:np.ndarray) -> np.ndarray:
    """Takes a 3-dimensional array representing an image in the YCbCr color
    space, and returns an array of the image in the RGB color space:
    array(width, heigth, YCbCr) -> array(width, heigth, RGB)
    """
    print("\nConverting colors from YCbCr to RGB...")
    Y = image_array[..., 0].astype("float64")
    Cb = image_array[..., 1].astype("float64")
    Cr = image_array[..., 2].astype("float64")

    R = Y + 1.402 * (Cr - 128.0)
    G = Y - 0.34414 * (Cb - 128.0) - 0.71414 * (Cr - 128.0)
    B = Y + 1.772 * (Cb - 128.0)

    output = np.stack((R, G, B), axis=-1)
    np.clip(output, a_min=0.0, a_max=255.0, out=output)

    return np.round(output).astype("uint8")

def print_progress(current:int, total:int, done:bool=False, header:str="Progress") -> None:
    """Print a progress percentage on the screen. If the process is not done yet,
    the line is updated instead of moving to the next line.
    """
    if not done:
        print(f"{header}: {current}/{total} ({current * 100 / total:.2f}%)", end="\r")
    else:
        print(f"{header}: {current}/{total} ({current * 100 / total:.0f}%) DONE!")

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
    from sys import argv
    try:
        from tkinter.filedialog import askopenfilename
        import tkinter as tk
        dialog = True
    except ModuleNotFoundError:
        dialog = False

    # Get the JPEG file path
    # If a path was provided as a command line argument, then use it
    if len(argv) > 1:
        jpeg_path = Path(argv[1])
        command = True
    else:
        command = False
    
    while True:
        
        # Open a dialog to ask the user for a image path
        if not command:
            if dialog:
                window = tk.Tk()
                window.state("withdrawn")
                print("Please choose a JPEG image...")
                jpeg_path = Path(
                    askopenfilename(
                        master = None,
                        title = "Decode a JPEG image",
                        filetypes = (
                            ("JPEG images", "*.jpg *.jpeg *.jfif *.jpe *.jif *.jfi"),
                            ("All files", "*.*")
                        )
                    )
                )
                window.destroy()
                
                # Check if the user has chosen something
                # (if the window was closed, then it returns an empty path)
                if jpeg_path == Path():
                    print("No file was selected.")
                    jpeg_path = None
                    break
            
            # If no GUI is available, then use the command prompt to ask the user for a path
            else:
                jpeg_path = Path(input("JPEG path: "))
        
        # Check if the provided path exists
        if jpeg_path.exists():
            break
        
        # Ask the user to try again if the path does not exist
        else:
            command = False
            print(f"File '{jpeg_path.name}' was not found on '{jpeg_path.parent.resolve()}'")
            
            # Ask yes or no
            while True:
                user_input = input("Try again with another file? [y]es / [n]o: ").lstrip().lower()[0]
                if user_input in "yn":
                    break
            
            # Break or continue the "get file path" loop
            if user_input == "y":
                continue
            elif user_input == "n":
                jpeg_path = None
                break

    
    # Decode the image
    if jpeg_path is not None:
        JpegDecoder(jpeg_path)
    print("Program finished. Have a nice day!")