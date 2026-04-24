import mmap
import os
import struct
import time

import numpy as np


# AXI lite register memory map, we mapped these to real offsets from the IP Base
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
LEFT_BASE_FRAME_ADDR = 0x30
RIGHT_BASE_FRAME_ADDR = 0x34
BASE_FRAME_VALID = 0x38
STATUS_BUSY_MASK = 0x1
STATUS_DONE_MASK = 0x2
STATUS_RESULT_VALID_MASK = 0x4

# IP Config
IP_BASE = 0x43C00000
IP_SIZE = 0x1000

# Image Config
IMAGE_SHAPE = (1080, 1920, 3)
IMAGE_DTYPE = np.uint8
# MATLAB/ TCP image contract is RGB
RED_CHANNEL_INDEX = 0


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
        timeout_s=15.0,
        poll_interval_s=0.001,
    ):
        if dma_engine is None:
            raise RuntimeError("dma_engine is required for PingPongFpgaCache")
        if not hasattr(dma_engine, "sendchannel_left") or not hasattr(dma_engine, "sendchannel_right"):
            raise TypeError("PingPongFpgaCache requires stereo DMA with sendchannel_left and sendchannel_right")
        if not hasattr(dma_engine.sendchannel_left, "transfer") or not hasattr(dma_engine.sendchannel_left, "wait"):
            raise TypeError("left DMA sendchannel must expose transfer and wait")
        if not hasattr(dma_engine.sendchannel_right, "transfer") or not hasattr(dma_engine.sendchannel_right, "wait"):
            raise TypeError("right DMA sendchannel must expose transfer and wait")

        self.image_shape = tuple(image_shape)
        self.fpga_image_shape = self.get_fpga_image_shape(self.image_shape)
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
        self.left_base_buffer, self.right_base_buffer = self.allocate_base_buffers()
        self.base_frame_valid = False
        self.last_submission_was_base_update = False
        self.open_registers()
        self.initialize_base_buffers()

    # Allocate two ping pong frame buffers that are FPGA visible
    def allocate_buffers(self):
        left_buffers = [
            self.allocate(shape=self.fpga_image_shape, dtype=self.image_dtype),
            self.allocate(shape=self.fpga_image_shape, dtype=self.image_dtype),
        ]
        right_buffers = [
            self.allocate(shape=self.fpga_image_shape, dtype=self.image_dtype),
            self.allocate(shape=self.fpga_image_shape, dtype=self.image_dtype),
        ]
        return left_buffers, right_buffers

    # Allocate persistent base frame buffers that the FPGA can reference
    def allocate_base_buffers(self):
        left_base_buffer = self.allocate(shape=self.fpga_image_shape, dtype=self.image_dtype)
        right_base_buffer = self.allocate(shape=self.fpga_image_shape, dtype=self.image_dtype)
        return left_base_buffer, right_base_buffer

    # FPGA fast path is processing only the red channel
    def get_fpga_image_shape(self, image_shape):
        if len(image_shape) >= 3:
            return image_shape[0], image_shape[1]
        return image_shape

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

    # Initialize base buffers to zero until frame 0 supplies the real base image
    def initialize_base_buffers(self):
        self.left_base_buffer.fill(0)
        self.right_base_buffer.fill(0)
        if hasattr(self.left_base_buffer, "flush"):
            self.left_base_buffer.flush()
        if hasattr(self.right_base_buffer, "flush"):
            self.right_base_buffer.flush()
        self.write_base_registers()

    # Copy an image into the current ping pong buffer
    def copy_into_buffer(self, buffer_obj, image_data):
        image_array = np.asarray(image_data, dtype=self.image_dtype)
        if image_array.shape == self.image_shape and len(self.image_shape) >= 3:
            image_array = np.ascontiguousarray(image_array[:, :, RED_CHANNEL_INDEX], dtype=self.image_dtype)
        elif image_array.shape == self.fpga_image_shape:
            image_array = np.ascontiguousarray(image_array, dtype=self.image_dtype)
        else:
            raise ValueError("image shape does not match configured FPGA buffer shape")

        np.copyto(buffer_obj, image_array, casting="safe")
        if hasattr(buffer_obj, "flush"):
            buffer_obj.flush()

    # Return the device address for the hardware backed buffer
    def buffer_address(self, buffer_obj):
        return int(buffer_obj.device_address)

    # Write the persistent base frame buffer addresses into the detector registers
    def write_base_registers(self):
        self.write_reg_u32(LEFT_BASE_FRAME_ADDR, self.buffer_address(self.left_base_buffer))
        self.write_reg_u32(RIGHT_BASE_FRAME_ADDR, self.buffer_address(self.right_base_buffer))
        if self.base_frame_valid:
            self.write_reg_u32(BASE_FRAME_VALID, 1)
        else:
            self.write_reg_u32(BASE_FRAME_VALID, 0)

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
        return (self.read_reg_u32(STATUS) & STATUS_BUSY_MASK) == STATUS_BUSY_MASK

    # Return whether the FPGA reports that a frame is fully processed
    def fpga_done(self):
        return (self.read_reg_u32(STATUS) & STATUS_DONE_MASK) == STATUS_DONE_MASK

    # Ensure the DMA MM2S channel is actually running before issuing a transfer
    def ensure_dma_channel_running(self, label, channel):
        if hasattr(channel, "running"):
            try:
                if not channel.running and hasattr(channel, "start"):
                    channel.start()
            except Exception:
                pass

    # Wait for the FPGA to finish the active frame
    def wait_until_idle(self, timeout_s=None):
        if timeout_s is None:
            timeout_s = self.timeout_s

        start_time = time.time()
        while self.fpga_busy():
            if time.time() - start_time > timeout_s:
                raise TimeoutError("Timed out waiting for FPGA to go idle")
            time.sleep(self.poll_interval_s)

    # Wait until the detector reports completion through the AXI-lite status register
    def wait_until_complete(self, timeout_s=None):
        if timeout_s is None:
            timeout_s = self.timeout_s

        start_time = time.time()
        while True:
            status = self.read_reg_u32(STATUS)
            if (status & STATUS_DONE_MASK) and not (status & STATUS_BUSY_MASK):
                return
            if time.time() - start_time > timeout_s:
                raise TimeoutError(
                    "Timed out waiting for FPGA detector completion; last STATUS=0x%08X" % status
                )
            time.sleep(self.poll_interval_s)

    # Start the AXI DMA processing path for the current stereo buffer pair
    def start_hardware_transfer(self, left_buffer, right_buffer, frame_number):
        self.write_reg_u32(LEFT_FRAME_ADDR, self.buffer_address(left_buffer))
        self.write_reg_u32(RIGHT_FRAME_ADDR, self.buffer_address(right_buffer))
        self.write_reg_u32(FRAME_WIDTH, self.fpga_image_shape[1])
        self.write_reg_u32(FRAME_HEIGHT, self.fpga_image_shape[0])
        self.write_reg_u32(FRAME_CHANS, 1)

        if frame_number is None:
            self.write_reg_u32(FRAME_ID, 0xFFFFFFFF)
        else:
            self.write_reg_u32(FRAME_ID, frame_number)
        self.write_reg_u32(BUFFER_INDEX, self.write_index)
        self.write_reg_u32(CTRL, 1)
        self.submit_dma(left_buffer, right_buffer)

    # Submit the current stereo buffers to the DMA engine
    def submit_dma(self, left_buffer, right_buffer):
        self.ensure_dma_channel_running("left", self.dma_engine.sendchannel_left)
        self.ensure_dma_channel_running("right", self.dma_engine.sendchannel_right)
        self.dma_engine.sendchannel_left.transfer(left_buffer)
        self.dma_engine.sendchannel_right.transfer(right_buffer)

    # Wait for DMA transfer to complete
    def wait_for_dma(self):
        self.dma_engine.sendchannel_left.wait()
        self.dma_engine.sendchannel_right.wait()

    # Store frame 0 as the persistent stereo base image for FPGA subtraction
    def update_base_frame(self, left_image, right_image):
        self.copy_into_buffer(self.left_base_buffer, left_image)
        self.copy_into_buffer(self.right_base_buffer, right_image)
        self.base_frame_valid = True
        self.write_base_registers()

    # Cache one frame in DDR and kick off FPGA processing when available
    def submit_frame(self, frame_number, image_data):
        if frame_number is None:
            frame_number = -1

        self.wait_until_idle()
        left_image, right_image = self.extract_stereo_images(image_data)
        self.last_submission_was_base_update = False

        if frame_number == 0:
            self.update_base_frame(left_image, right_image)
            self.last_frame_number = frame_number
            self.last_submission_was_base_update = True
            return

        left_buffer, right_buffer = self.current_buffer_pair()
        self.copy_into_buffer(left_buffer, left_image)
        self.copy_into_buffer(right_buffer, right_image)

        self.last_frame_number = frame_number
        self.last_submitted_index = self.write_index
        self.start_hardware_transfer(left_buffer, right_buffer, frame_number)
        self.write_index = 1 - self.write_index

    # Read the latest FPGA output after the DMA transfer completes
    def read_result(self):
        if self.last_submission_was_base_update:
            return {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "base_updated": True,
            }

        self.wait_until_complete()
        status = self.read_reg_u32(STATUS)
        left_x = self.read_reg_f32(RESULT_X)
        left_y = self.read_reg_f32(RESULT_Y)
        right_x = self.read_reg_f32(RESULT_Z)

        return {
            "x": left_x,
            "y": left_y,
            "z": right_x,
            "left_x": left_x,
            "left_y": left_y,
            "right_x": right_x,
            "candidate_valid": (status & STATUS_RESULT_VALID_MASK) == STATUS_RESULT_VALID_MASK,
            "status": status,
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

        for buffer_obj in [self.left_base_buffer, self.right_base_buffer]:
            try:
                buffer_obj.close()
            except AttributeError:
                pass
        self.left_base_buffer = None
        self.right_base_buffer = None

        if self.regs is not None:
            self.regs.close()
            self.regs = None

        if self.mem_file is not None:
            self.mem_file.close()
            self.mem_file = None
