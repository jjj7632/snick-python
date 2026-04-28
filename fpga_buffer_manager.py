import mmap
import os
import struct
import time
from pathlib import Path

import cv2
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
UNKNOWN_FRAME_NUMBER = 0xFFFFFFFF
STATUS_BIT_NAMES = {
    0: "busy",
    1: "done",
    2: "result_valid",
    3: "base_valid",
    4: "left_live_word",
    5: "left_base_word",
    6: "right_live_word",
    7: "right_base_word",
    8: "left_done",
    9: "right_done",
    10: "left_mask_valid",
    11: "right_mask_valid",
    12: "left_base_reader_busy",
    13: "left_base_reader_done",
    14: "right_base_reader_busy",
    15: "right_base_reader_done",
    16: "left_base_stream_valid",
    17: "left_base_stream_ready",
    18: "right_base_stream_valid",
    19: "right_base_stream_ready",
    20: "left_base_arvalid",
    21: "left_base_arready",
    22: "left_base_rvalid",
    23: "left_base_rready",
    24: "right_base_arvalid",
    25: "right_base_arready",
    26: "right_base_rvalid",
    27: "right_base_rready",
    28: "left_live_tvalid",
    29: "left_live_tready",
    30: "right_live_tvalid",
    31: "right_live_tready",
}

# IP Config
IP_BASE = 0x43C00000
IP_SIZE = 0x1000

# Image Config
IMAGE_SHAPE = (1080, 1920, 3)
IMAGE_DTYPE = np.uint8
# Live MATLAB frames arrive as RGB so the physical red channel is index 0 here
LIVE_RED_CHANNEL_INDEX = 0
CV2_RED_CHANNEL_INDEX = 2
FAST_PRELOAD_BASE_FRAME = 40


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
        if not hasattr(dma_engine, "recvchannel_left_mask") or not hasattr(dma_engine, "recvchannel_right_mask"):
            raise TypeError(
                "PingPongFpgaCache requires stereo DMA with recvchannel_left_mask and recvchannel_right_mask"
            )
        if not hasattr(dma_engine.sendchannel_left, "transfer") or not hasattr(dma_engine.sendchannel_left, "wait"):
            raise TypeError("left DMA sendchannel must expose transfer and wait")
        if not hasattr(dma_engine.sendchannel_right, "transfer") or not hasattr(dma_engine.sendchannel_right, "wait"):
            raise TypeError("right DMA sendchannel must expose transfer and wait")
        if not hasattr(dma_engine.recvchannel_left_mask, "transfer") or not hasattr(dma_engine.recvchannel_left_mask, "wait"):
            raise TypeError("left mask DMA recvchannel must expose transfer and wait")
        if not hasattr(dma_engine.recvchannel_right_mask, "transfer") or not hasattr(dma_engine.recvchannel_right_mask, "wait"):
            raise TypeError("right mask DMA recvchannel must expose transfer and wait")

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
        (
            self.base_buffer_owner,
            self.left_base_buffer,
            self.right_base_buffer,
            self.left_base_address,
            self.right_base_address,
        ) = self.allocate_base_buffers()
        self.left_mask_buffer, self.right_mask_buffer = self.allocate_mask_buffers()
        self.base_frame_valid = False
        self.last_submission_was_base_update = False
        self.last_live_addresses = None
        self.last_mask_addresses = None
        self.open_registers()
        self.reset_dma_channels()
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
        owner = self.allocate(shape=(2,) + self.fpga_image_shape, dtype=self.image_dtype)
        frame_bytes = int(np.prod(self.fpga_image_shape)) * self.image_dtype.itemsize
        left_base_buffer = owner[0]
        right_base_buffer = owner[1]
        left_base_address = int(owner.device_address)
        right_base_address = left_base_address + frame_bytes
        return owner, left_base_buffer, right_base_buffer, left_base_address, right_base_address

    # Allocate persistent output buffers for the thresholded masks written back by the FPGA
    def allocate_mask_buffers(self):
        left_mask_buffer = self.allocate(shape=self.fpga_image_shape, dtype=self.image_dtype)
        right_mask_buffer = self.allocate(shape=self.fpga_image_shape, dtype=self.image_dtype)
        left_mask_buffer.fill(0)
        right_mask_buffer.fill(0)
        if hasattr(left_mask_buffer, "flush"):
            left_mask_buffer.flush()
        if hasattr(right_mask_buffer, "flush"):
            right_mask_buffer.flush()
        return left_mask_buffer, right_mask_buffer

    # FPGA fast path is processing only the red channel
    def get_fpga_image_shape(self, image_shape):
        if len(image_shape) >= 3:
            return image_shape[0], image_shape[1]
        return image_shape

    # Return the expected byte count for one one channel FPGA frame
    def expected_frame_bytes(self):
        return int(np.prod(self.fpga_image_shape)) * self.image_dtype.itemsize

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

    # Initialize base buffers from the tuned preload image when available
    def initialize_base_buffers(self):
        if self.try_preload_base_buffers():
            return
        self.left_base_buffer.fill(0)
        self.right_base_buffer.fill(0)
        if hasattr(self.base_buffer_owner, "flush"):
            self.base_buffer_owner.flush()
        self.write_base_registers()

    # Try to preload the same frame 40 stereo base used by the software FAST path
    def try_preload_base_buffers(self):
        repo_root = Path(__file__).resolve().parents[1]
        left_path = None
        right_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate_left = repo_root / "serve1" / ("LeftFrame_%d%s" % (FAST_PRELOAD_BASE_FRAME, ext))
            candidate_right = repo_root / "serve1" / ("RightFrame_%d%s" % (FAST_PRELOAD_BASE_FRAME, ext))
            if candidate_left.is_file() and candidate_right.is_file():
                left_path = candidate_left
                right_path = candidate_right
                break

        if left_path is None or right_path is None:
            return False

        left_image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_image = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_image is None or right_image is None:
            return False

        np.copyto(self.left_base_buffer, np.ascontiguousarray(left_image[:, :, CV2_RED_CHANNEL_INDEX], dtype=self.image_dtype), casting="safe")
        np.copyto(self.right_base_buffer, np.ascontiguousarray(right_image[:, :, CV2_RED_CHANNEL_INDEX], dtype=self.image_dtype), casting="safe")
        if hasattr(self.base_buffer_owner, "flush"):
            self.base_buffer_owner.flush()
        self.base_frame_valid = True
        self.write_base_registers()
        return True

    # Copy an image into the current ping pong buffer
    def copy_into_buffer(self, buffer_obj, image_data):
        image_array = np.asarray(image_data, dtype=self.image_dtype)
        if image_array.shape == self.image_shape and len(self.image_shape) >= 3:
            image_array = np.ascontiguousarray(image_array[:, :, LIVE_RED_CHANNEL_INDEX], dtype=self.image_dtype)
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
        self.write_reg_u32(LEFT_BASE_FRAME_ADDR, self.left_base_address)
        self.write_reg_u32(RIGHT_BASE_FRAME_ADDR, self.right_base_address)
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
                running = channel.running
                if not running and hasattr(channel, "start"):
                    channel.start()
            except Exception:
                pass

    # FPGA channels get left in a weird state found from debugging, resetting them in init
    def reset_dma_channels(self):
        for label, channel in [
            ("left", self.dma_engine.sendchannel_left),
            ("right", self.dma_engine.sendchannel_right),
            ("left_mask", self.dma_engine.recvchannel_left_mask),
            ("right_mask", self.dma_engine.recvchannel_right_mask),
        ]:
            mmio_obj = getattr(channel, "_mmio", None)
            offset = getattr(channel, "_offset", None)
            if mmio_obj is None or offset is None:
                continue
            # Issue soft reset via DMACR bit 2
            mmio_obj.write(offset, 0x00000004)
            import time
            time.sleep(0.05)
            # Set RS=1 to start the channel
            mmio_obj.write(offset, 0x00000001)

    # Best effort snapshot of a DMA channel to make timeout diagnosis much easier
    def snapshot_dma_channel(self, label, channel):
        state = {
            "label": label,
            "running": None,
            "idle": None,
        }

        try:
            state["running"] = bool(channel.running)
        except Exception:
            pass

        try:
            state["idle"] = bool(channel.idle)
        except Exception:
            pass

        mmio = getattr(channel, "_mmio", None)
        offset = getattr(channel, "_offset", None)
        if mmio is not None and offset is not None:
            try:
                dmacr = int(mmio.read(offset))
                dmasr = int(mmio.read(offset + 0x04))
                state["dmacr"] = "0x%08X" % dmacr
                state["dmasr"] = "0x%08X" % dmasr
                state["halted"] = bool(dmasr & 0x00000001)
                state["idle_bit"] = bool(dmasr & 0x00000002)
                state["dma_int_err"] = bool(dmasr & 0x00000010)
                state["dma_slv_err"] = bool(dmasr & 0x00000020)
                state["dma_dec_err"] = bool(dmasr & 0x00000040)
                state["sg_int_err"] = bool(dmasr & 0x00000100)
                state["sg_slv_err"] = bool(dmasr & 0x00000200)
                state["sg_dec_err"] = bool(dmasr & 0x00000400)
                state["ioc_irq"] = bool(dmasr & 0x00001000)
                state["dly_irq"] = bool(dmasr & 0x00002000)
                state["err_irq"] = bool(dmasr & 0x00004000)
            except Exception:
                pass

        return state

    def snapshot_all_dma_channels(self):
        return {
            "left_live": self.snapshot_dma_channel("left_live", self.dma_engine.sendchannel_left),
            "right_live": self.snapshot_dma_channel("right_live", self.dma_engine.sendchannel_right),
            "left_mask": self.snapshot_dma_channel("left_mask", self.dma_engine.recvchannel_left_mask),
            "right_mask": self.snapshot_dma_channel("right_mask", self.dma_engine.recvchannel_right_mask),
        }

    def decode_status_bits(self, status):
        active = []
        for bit_index, name in sorted(STATUS_BIT_NAMES.items()):
            if status & (1 << bit_index):
                active.append(name)
        return active

    def build_timeout_context(self, status):
        reg_snapshot = {
            "frame_width": self.read_reg_u32(FRAME_WIDTH),
            "frame_height": self.read_reg_u32(FRAME_HEIGHT),
            "frame_chans": self.read_reg_u32(FRAME_CHANS),
            "left_base_reg": "0x%08X" % self.read_reg_u32(LEFT_BASE_FRAME_ADDR),
            "right_base_reg": "0x%08X" % self.read_reg_u32(RIGHT_BASE_FRAME_ADDR),
            "base_valid_reg": self.read_reg_u32(BASE_FRAME_VALID),
        }
        return {
            "status_hex": "0x%08X" % status,
            "status_bits": self.decode_status_bits(status),
            "detector_regs": reg_snapshot,
            "live_addrs": self.last_live_addresses,
            "base_addrs": {
                "left_base": "0x%08X" % self.left_base_address,
                "right_base": "0x%08X" % self.right_base_address,
            },
            "mask_addrs": self.last_mask_addresses,
            "dma": self.snapshot_all_dma_channels(),
        }

    # Wait for the FPGA to finish the active frame
    def wait_until_idle(self, timeout_s=None):
        if timeout_s is None:
            timeout_s = self.timeout_s

        start_time = time.time()
        while self.fpga_busy():
            if time.time() - start_time > timeout_s:
                status = self.read_reg_u32(STATUS)
                raise TimeoutError(
                    "Timed out waiting for FPGA to go idle; context=%s" % self.build_timeout_context(status)
                )
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
                    "Timed out waiting for FPGA detector completion; context=%s" % self.build_timeout_context(status)
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
            self.write_reg_u32(FRAME_ID, UNKNOWN_FRAME_NUMBER)
        else:
            self.write_reg_u32(FRAME_ID, frame_number)
        self.write_reg_u32(BUFFER_INDEX, self.write_index)
        self.last_live_addresses = {
            "left_live": "0x%08X" % self.buffer_address(left_buffer),
            "right_live": "0x%08X" % self.buffer_address(right_buffer),
        }
        self.last_mask_addresses = {
            "left_mask": "0x%08X" % self.buffer_address(self.left_mask_buffer),
            "right_mask": "0x%08X" % self.buffer_address(self.right_mask_buffer),
        }
        self.start_mask_dma()
        self.write_reg_u32(CTRL, 1)        # arm IP first
        self.submit_dma(left_buffer, right_buffer)  # then start streams

    # Submit the current stereo buffers to the DMA engine
    def submit_dma(self, left_buffer, right_buffer):
        left_transfer_view = left_buffer.reshape(-1)
        right_transfer_view = right_buffer.reshape(-1)
        self.ensure_dma_channel_running("left", self.dma_engine.sendchannel_left)
        self.ensure_dma_channel_running("right", self.dma_engine.sendchannel_right)
        self.dma_engine.sendchannel_left.transfer(left_transfer_view)
        self.dma_engine.sendchannel_right.transfer(right_transfer_view)

    # Arm the S2MM channels that write the detector's binary masks back into DDR
    def start_mask_dma(self):
        left_mask_transfer_view = self.left_mask_buffer.reshape(-1)
        right_mask_transfer_view = self.right_mask_buffer.reshape(-1)
        self.ensure_dma_channel_running("left_mask", self.dma_engine.recvchannel_left_mask)
        self.ensure_dma_channel_running("right_mask", self.dma_engine.recvchannel_right_mask)
        self.dma_engine.recvchannel_left_mask.transfer(left_mask_transfer_view)
        self.dma_engine.recvchannel_right_mask.transfer(right_mask_transfer_view)

    # Wait for DMA transfer to complete
    def wait_for_dma(self):
        self.dma_engine.sendchannel_left.wait()
        self.dma_engine.sendchannel_right.wait()

    # Wait for the returned mask streams to be committed into DDR
    def wait_for_mask_dma(self):
        self.dma_engine.recvchannel_left_mask.wait()
        self.dma_engine.recvchannel_right_mask.wait()

    # Store frame 0 as the persistent stereo base image for FPGA subtraction
    def update_base_frame(self, left_image, right_image):
        self.copy_into_buffer(self.left_base_buffer, left_image)
        self.copy_into_buffer(self.right_base_buffer, right_image)
        if hasattr(self.base_buffer_owner, "flush"):
            self.base_buffer_owner.flush()
        self.base_frame_valid = True
        self.write_base_registers()

    # Cache one frame in DDR and kick off FPGA processing when available
    def submit_frame(self, frame_number, image_data):
        self.wait_until_idle()
        left_image, right_image = self.extract_stereo_images(image_data)
        self.last_submission_was_base_update = False

        if (not self.base_frame_valid) or frame_number == 0:
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
                "base_updated": True,
            }

        self.wait_for_dma()
        self.wait_until_complete()
        self.wait_for_mask_dma()
        if hasattr(self.left_mask_buffer, "invalidate"):
            self.left_mask_buffer.invalidate()
        if hasattr(self.right_mask_buffer, "invalidate"):
            self.right_mask_buffer.invalidate()
        status = self.read_reg_u32(STATUS)
        result_x = self.read_reg_f32(RESULT_X)
        result_y = self.read_reg_f32(RESULT_Y)
        result_z = self.read_reg_f32(RESULT_Z)
        return {
            "base_updated": False,
            "left_mask": np.asarray(self.left_mask_buffer).copy(),
            "right_mask": np.asarray(self.right_mask_buffer).copy(),
            "candidate_valid": (status & STATUS_RESULT_VALID_MASK) == STATUS_RESULT_VALID_MASK,
            "status": status,
            "result_x": result_x,
            "result_y": result_y,
            "result_z": result_z,
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

        if self.base_buffer_owner is not None:
            try:
                self.base_buffer_owner.close()
            except AttributeError:
                pass
        self.base_buffer_owner = None
        self.left_base_buffer = None
        self.right_base_buffer = None

        for buffer_obj in [self.left_mask_buffer, self.right_mask_buffer]:
            try:
                buffer_obj.close()
            except AttributeError:
                pass
        self.left_mask_buffer = None
        self.right_mask_buffer = None

        if self.regs is not None:
            self.regs.close()
            self.regs = None

        if self.mem_file is not None:
            self.mem_file.close()
            self.mem_file = None
