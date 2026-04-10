import mmap
import os
import struct
import time

import numpy as np

# Saved me a headache, serves like a macro
try:
    from pynq import allocate
except ImportError:
    allocate = None


# AXI lite register memory map, we need to map these to real offsets from the IP Base
CTRL = 0x00
STATUS = 0x04
FRAME_ADDR = 0x08
FRAME_WIDTH = 0x0C
FRAME_HEIGHT = 0x10
FRAME_CHANS = 0x14
FRAME_ID = 0x18
BUFFER_INDEX = 0x1C

# Result in Memory
RESULT_X = 0x20
RESULT_Y = 0x24
RESULT_Z = 0x28

# IP Config
IP_BASE = 0x43C60000    
IP_SIZE = 0x1000

# Image Config
IMAGE_SHAPE = (1080, 1920, 3)
IMAGE_DTYPE = np.uint8

class PingPongFpgaCache(object):
    # Create FPGA visible frame buffers and required AXI register access
    def __init__(
        self,
        dma_engine,
        image_shape=IMAGE_SHAPE,
        image_dtype=IMAGE_DTYPE,
        ip_base=IP_BASE,
        ip_size=IP_SIZE,
        mem_path="/dev/mem",
        timeout_s=1.0,
        poll_interval_s=0.001,
    ):
        if allocate is None:
            raise RuntimeError("pynq.allocate is required for the DMA frame buffer path")
        if dma_engine is None:
            raise RuntimeError("dma_engine is required for PingPongFpgaCache")

        self.image_shape = tuple(image_shape)
        self.image_dtype = np.dtype(image_dtype)
        self.ip_base = ip_base
        self.ip_size = ip_size
        self.mem_path = mem_path
        self.dma_engine = dma_engine
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.write_index = 0
        self.last_submitted_index = None
        self.last_frame_number = None
        self.mem_file = None
        self.regs = None
        self.buffers = self.allocate_buffers()
        self.open_registers()

    # Allocate two ping pong frame buffers that are FPGA visible
    def allocate_buffers(self):
        return [
            allocate(shape=self.image_shape, dtype=self.image_dtype),
            allocate(shape=self.image_shape, dtype=self.image_dtype),
        ]

    # Open the AXI register window on hardware
    def open_registers(self):
        if not os.path.exists(self.mem_path):
            raise RuntimeError("%s is required for AXI register access" % self.mem_path)

        try:
            self.mem_file = open(self.mem_path, "r+b")
            self.regs = mmap.mmap(self.mem_file.fileno(), self.ip_size, offset=self.ip_base)
        except OSError:
            self.regs = None
            if self.mem_file is not None:
                self.mem_file.close()
                self.mem_file = None

    # Return the active frame buffer that Python should fill next
    def current_buffer(self):
        return self.buffers[self.write_index]

    # Copy an image into the current ping pong buffer
    def copy_into_buffer(self, buffer_obj, image_data):
        image_array = np.asarray(image_data, dtype=self.image_dtype)
        if image_array.shape != self.image_shape:
            raise ValueError("image shape does not match configured FPGA buffer shape")

        np.copyto(buffer_obj, image_array, casting="safe")
        if hasattr(buffer_obj, "flush"):
            buffer_obj.flush()

    # Return the device address for a hardware backed buffer
    def buffer_address(self, buffer_obj):
        return int(buffer_obj.device_address)

    # Write one unsigned AXI register value
    def write_reg_u32(self, offset, value):
        self.regs.seek(offset)
        packed_value = struct.pack("<I", int(value) & 0xFFFFFFFF)
        self.regs.write(packed_value)

    # Read one unsigned AXI register value
    def read_reg_u32(self, offset):
        self.regs.seek(offset)
        data = self.regs.read(4)
        value = struct.unpack("<I", data)[0]
        return value

    # Read one floating point AXI register value
    def read_reg_f32(self, offset):
        self.regs.seek(offset)
        data = self.regs.read(4)
        value = struct.unpack("<f", data)[0]
        return value

    # Return whether the FPGA status register reports a busy engine
    def fpga_busy(self):
        return (self.read_reg_u32(STATUS) & 0x1) == 1

    # Wait for the FPGA to finish the active frame
    def wait_until_idle(self, timeout_s=None):
        if timeout_s is None:
            timeout_s = self.timeout_s

        start_time = time.time()
        while self.fpga_busy():
            if time.time() - start_time > timeout_s:
                raise TimeoutError("Timed out waiting for FPGA to go idle")
            time.sleep(self.poll_interval_s)

    # Start the AXI DMA processing path for the current frame buffer
    def start_hardware_transfer(self, buffer_obj, frame_number):
        self.write_reg_u32(FRAME_ADDR, self.buffer_address(buffer_obj))
        self.write_reg_u32(FRAME_WIDTH, self.image_shape[1])
        self.write_reg_u32(FRAME_HEIGHT, self.image_shape[0])

        if len(self.image_shape) >= 3:
            channels = self.image_shape[2]
        else:
            channels = 1
        self.write_reg_u32(FRAME_CHANS, channels)

        if frame_number is None:
            self.write_reg_u32(FRAME_ID, 0xFFFFFFFF)
        else:
            self.write_reg_u32(FRAME_ID, frame_number)
        self.write_reg_u32(BUFFER_INDEX, self.write_index)
        self.submit_dma(buffer_obj)
        self.write_reg_u32(CTRL, 1)

    # Submit the current buffer to the required DMA engine
    def submit_dma(self, buffer_obj):
        if hasattr(self.dma_engine, "sendchannel"):
            self.dma_engine.sendchannel.transfer(buffer_obj)
            return

        if hasattr(self.dma_engine, "transfer"):
            self.dma_engine.transfer(buffer_obj)
            return

        raise TypeError("Unsupported DMA engine interface")

    # Wait for DMA transfer to complete
    def wait_for_dma(self):
        if hasattr(self.dma_engine, "sendchannel"):
            self.dma_engine.sendchannel.wait()
            return

        if hasattr(self.dma_engine, "wait"):
            self.dma_engine.wait()
            return

    # Cache one frame in DDR and kick off FPGA processing when available
    def submit_frame(self, frame_number, image_data):
        if frame_number is None:
            frame_number = -1

        self.wait_until_idle()
        buffer_obj = self.current_buffer()
        self.copy_into_buffer(buffer_obj, image_data)

        self.last_frame_number = frame_number
        self.last_submitted_index = self.write_index
        self.start_hardware_transfer(buffer_obj, frame_number)
        self.write_index = 1 - self.write_index

    # Read the latest FPGA output after the DMA transfer completes
    def read_result(self):
        self.wait_for_dma()
        self.wait_until_idle()

        return {
            "x": self.read_reg_f32(RESULT_X),
            "y": self.read_reg_f32(RESULT_Y),
            "z": self.read_reg_f32(RESULT_Z),
        }

    # Release allocated frame buffers and AXI register mappings
    def close(self):
        for buffer_obj in self.buffers:
            try:
                buffer_obj.close()
            except AttributeError:
                pass
        self.buffers = []

        if self.regs is not None:
            self.regs.close()
            self.regs = None

        if self.mem_file is not None:
            self.mem_file.close()
            self.mem_file = None
