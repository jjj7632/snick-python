import mmap
import os
import struct
import time

import numpy as np


# AXI lite register memory map, we need to map these to real offsets from the IP Base
CTRL = 0x00
STATUS = 0x04
LEFT_FRAME_ADDR = 0x08
RIGHT_FRAME_ADDR = 0x0C
FRAME_WIDTH = 0x10
FRAME_HEIGHT = 0x14
FRAME_CHANS = 0x18
FRAME_ID = 0x1C
BUFFER_INDEX = 0x20


# Result in Memory
RESULT_X = 0x24
RESULT_Y = 0x28
RESULT_Z = 0x2C

# IP Config
IP_BASE = 0x43C60000    
IP_SIZE = 0x1000

# Image Config
IMAGE_SHAPE = (1080, 1920, 3)
IMAGE_DTYPE = np.uint8


def get_pynq_allocate():
    try:
        from pynq import allocate
    except Exception as exc:
        raise RuntimeError("pynq.allocate is required for the DMA frame buffer path") from exc
    return allocate


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
        self.allocate = get_pynq_allocate()
        self.left_buffers, self.right_buffers = self.allocate_buffers()
        self.open_registers()

    # Allocate two ping pong frame buffers that are FPGA visible
    def allocate_buffers(self):
        left_buffers = [
            self.allocate(shape=self.image_shape, dtype=self.image_dtype),
            self.allocate(shape=self.image_shape, dtype=self.image_dtype),
        ]
        right_buffers = [
            self.allocate(shape=self.image_shape, dtype=self.image_dtype),
            self.allocate(shape=self.image_shape, dtype=self.image_dtype),
        ]
        return left_buffers, right_buffers

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
            raise

    # Return the active stereo buffer pair that Python should fill next
    def current_buffer_pair(self):
        return self.left_buffers[self.write_index], self.right_buffers[self.write_index]

    # Copy an image into the current ping pong buffer
    def copy_into_buffer(self, buffer_obj, image_data):
        image_array = np.asarray(image_data, dtype=self.image_dtype)
        if image_array.shape != self.image_shape:
            raise ValueError("image shape does not match configured FPGA buffer shape")

        np.copyto(buffer_obj, image_array, casting="safe")
        if hasattr(buffer_obj, "flush"):
            buffer_obj.flush()

    # Return the device address for the hardware backed buffer
    def buffer_address(self, buffer_obj):
        return int(buffer_obj.device_address)

    # Return a left and right image from the stereo image data
    def extract_stereo_images(self, image_data):
        if isinstance(image_data, dict):
            # strict contract that image data will come in a dict with these key value pairs
            if "left_image" in image_data and "right_image" in image_data:
                return image_data["left_image"], image_data["right_image"]

        raise ValueError("image_data must be a stereo dict with left_image and right_image")

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

    # Start the AXI DMA processing path for the current stereo buffer pair
    def start_hardware_transfer(self, left_buffer, right_buffer, frame_number):
        self.write_reg_u32(LEFT_FRAME_ADDR, self.buffer_address(left_buffer))
        self.write_reg_u32(RIGHT_FRAME_ADDR, self.buffer_address(right_buffer))
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
        self.submit_dma(left_buffer, right_buffer)
        self.write_reg_u32(CTRL, 1)

    # Submit the current stereo buffers to the required DMA engine
    def submit_dma(self, left_buffer, right_buffer):
        if hasattr(self.dma_engine, "sendchannel_left") and hasattr(self.dma_engine, "sendchannel_right"):
            self.dma_engine.sendchannel_left.transfer(left_buffer)
            self.dma_engine.sendchannel_right.transfer(right_buffer)
            return

        if hasattr(self.dma_engine, "sendchannel"):
            self.dma_engine.sendchannel.transfer(left_buffer)
            self.dma_engine.sendchannel.wait()
            self.dma_engine.sendchannel.transfer(right_buffer)
            return

        if hasattr(self.dma_engine, "transfer"):
            self.dma_engine.transfer(left_buffer, right_buffer)
            return

        raise TypeError("Unsupported DMA engine interface")

    # Wait for DMA transfer to complete
    def wait_for_dma(self):
        if hasattr(self.dma_engine, "sendchannel_left") and hasattr(self.dma_engine, "sendchannel_right"):
            self.dma_engine.sendchannel_left.wait()
            self.dma_engine.sendchannel_right.wait()
            return

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
        left_image, right_image = self.extract_stereo_images(image_data)
        left_buffer, right_buffer = self.current_buffer_pair()
        self.copy_into_buffer(left_buffer, left_image)
        self.copy_into_buffer(right_buffer, right_image)

        self.last_frame_number = frame_number
        self.last_submitted_index = self.write_index
        self.start_hardware_transfer(left_buffer, right_buffer, frame_number)
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
        for buffer_obj in self.left_buffers:
            try:
                buffer_obj.close()
            except AttributeError:
                pass

        for buffer_obj in self.right_buffers:
            try:
                buffer_obj.close()
            except AttributeError:
                pass
        self.left_buffers = []
        self.right_buffers = []

        if self.regs is not None:
            self.regs.close()
            self.regs = None

        if self.mem_file is not None:
            self.mem_file.close()
            self.mem_file = None
