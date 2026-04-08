import mmap
import struct
import time
import numpy as np
from pynq import allocate

# AXI-Lite register map  **use the register map for the FPGA? to help finish**
CTRL          = 0x00
STATUS        = 0x04
FRAME_ADDR    = 0x08
FRAME_WIDTH   = 0x0C
FRAME_HEIGHT  = 0x10
FRAME_CHANS   = 0x14
FRAME_ID      = 0x18
BUFFER_INDEX  = 0x1C

RESULT_X      = 0x20
RESULT_Y      = 0x24
RESULT_Z      = 0x28

IP_BASE = 0x43C60000
IP_SIZE = 0x1000

IMAGE_SHAPE = (1080, 1920, 3)
IMAGE_DTYPE = np.uint8


class PingPongFpgaCache(object):
    def __init__(self):
        # These are FPGA accessible DDR buffers, Read the pynq allocate documentation for more info
        self.buffers = [
            allocate(shape=IMAGE_SHAPE, dtype=IMAGE_DTYPE),
            allocate(shape=IMAGE_SHAPE, dtype=IMAGE_DTYPE),
        ]

    def submit_frame(self, frame_number, image_data):
        if frame_number is None:
            return