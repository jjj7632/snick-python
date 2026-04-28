# -*- coding: utf-8 -*-
"""
SoC side TCP server for MATLAB/Python integration
Incoming commands are expected to use the binary layouts in numpysocket.py and
carry the command array protocol defined in soc_protocol.py which was outlined by August.
"""

import argparse
import os
import signal
import struct
import sys

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fpga_buffer_manager import PingPongFpgaCache
from shared_protocol.numpysocket import NumpySocket
from shared_protocol.soc_protocol import (
    CMD_LOG_DATA,
    CMD_PROCESS_IMAGE,
    CMD_REQUEST_IMAGE_AT_FRAME,
    CMD_REQUEST_LATEST_IMAGE,
    CMD_REQUEST_NTH_NEXT_IMAGE,
    CMD_REQUEST_NTH_PREVIOUS_IMAGE,
    CMD_RESET,
    CMD_SEND_CALL,
    CMD_SLAVE_MODE,
    CMD_SLAVE_MODE_READY,
    CMD_STOP_CAPTURE,
    SoCProtocol,
)

DEFAULT_OVERLAY_BIT = ""
DEFAULT_LEFT_DMA_NAME = "axi_dma_left"
DEFAULT_RIGHT_DMA_NAME = "axi_dma_right"
DEFAULT_LEFT_MASK_DMA_NAME = "axi_dma_left_mask"
DEFAULT_RIGHT_MASK_DMA_NAME = "axi_dma_right_mask"
DEFAULT_FPGA_IP_BASE = 0x43C00000
DEFAULT_FPGA_IP_SIZE = 0x1000
UNKNOWN_FRAME_NUMBER = 0xFFFFFFFF
COMMANDS_WITHOUT_ARGS = {
    CMD_REQUEST_LATEST_IMAGE,
    CMD_RESET,
    CMD_SLAVE_MODE,
    CMD_SLAVE_MODE_READY,
    CMD_STOP_CAPTURE,
}

COMMANDS_WITH_ONE_SIGNED_INT = {
    CMD_REQUEST_NTH_NEXT_IMAGE,
    CMD_REQUEST_NTH_PREVIOUS_IMAGE,
}

COMMANDS_WITH_ONE_UINT = {
    CMD_REQUEST_IMAGE_AT_FRAME,
}


class StereoDmaEngine(object):
    # Small adapter that lets the FPGA cache treat the live stereo DMAs as one engine
    def __init__(self, left_dma, right_dma, left_mask_dma, right_mask_dma):
        if not hasattr(left_dma, "sendchannel"):
            raise TypeError("left DMA object must expose sendchannel")
        if not hasattr(right_dma, "sendchannel"):
            raise TypeError("right DMA object must expose sendchannel")
        if not hasattr(left_mask_dma, "recvchannel"):
            raise TypeError("left mask DMA object must expose recvchannel")
        if not hasattr(right_mask_dma, "recvchannel"):
            raise TypeError("right mask DMA object must expose recvchannel")

        self.left_dma = left_dma
        self.right_dma = right_dma
        self.left_mask_dma = left_mask_dma
        self.right_mask_dma = right_mask_dma
        self.sendchannel_left = left_dma.sendchannel
        self.sendchannel_right = right_dma.sendchannel
        self.recvchannel_left_mask = left_mask_dma.recvchannel
        self.recvchannel_right_mask = right_mask_dma.recvchannel


class MatlabServerAdapter(object):
    # Set up the TCP server and bind it to a SoC protocol instance
    def __init__(
        self,
        host="",
        port=9999,
        image_shape=None,
        protocol=None,
        fpga_cache=None,
        disable_hardware=False,
        disable_fast_path=False,
        disable_fpga_fast_path=None,
    ):
        self.host = host
        self.port = port
        if image_shape is None:
            image_shape = (1080, 1920, 3)
        self.image_shape = tuple(image_shape)
        self.socket = NumpySocket(image_shape=self.image_shape)
        self.fpga_cache = fpga_cache
        if disable_fpga_fast_path is not None:
            disable_fast_path = bool(disable_fpga_fast_path)
        self.protocol = protocol or SoCProtocol(
            command_sender=self.send_soc_command,
            fpga_cache=self.fpga_cache,
            disable_hardware=disable_hardware,
            disable_fast_path=disable_fast_path,
        )
        self.protocol.command_sender = self.send_soc_command
        self.protocol.fpga_cache = self.fpga_cache
        self.is_running = False

    # Accept an incoming MATLAB or PC TCP client connection
    def start(self):
        self.socket.startServer(self.port, self.host)
        self.is_running = True

    # Stop serving and close any active socket connection
    def close(self):
        self.is_running = False
        self.socket.close()
        if self.fpga_cache is not None:
            try:
                self.fpga_cache.close()
            except AttributeError:
                pass
            self.fpga_cache = None

    # Process incoming commands until the client disconnects or a limit is reached
    def serve(self, max_packets=None):
        handled_packets = 0
        try:
            self.start()
            print("Connected to MATLAB client at %s:%s" % self.socket.client_address)
            while self.is_running:
                if max_packets is not None and handled_packets >= max_packets:
                    break

                self.protocol.drive()

                if not self.handle_next_command():
                    break

                handled_packets += 1
        finally:
            self.close()

    # Run the adapter without a packet limit
    def serve_forever(self):
        self.serve()

    # Handle one incoming command and execute it through the SoC protocol
    def handle_next_command(self):
        command = self.read_command()
        if command is None:
            return False

        try:
            self.protocol.handle_incoming_command(command)
        except Exception as exc:
            print("Command handling error:", exc)
        return True

    # Read one complete command from the socket using the agreed binary format
    def read_command(self):
        cmd = self.socket.receiveCmd()
        if cmd is None:
            return None

        if cmd in COMMANDS_WITHOUT_ARGS:
            return [cmd]

        if cmd in COMMANDS_WITH_ONE_SIGNED_INT:
            return [cmd, self.socket.receiveInt32()]

        if cmd in COMMANDS_WITH_ONE_UINT:
            return [cmd, self.restore_frame_number(self.socket.receiveUint32())]

        if cmd == CMD_SEND_CALL:
            return [cmd, self.socket.receiveUint8()]

        if cmd == CMD_LOG_DATA:
            frame_number = self.restore_frame_number(self.socket.receiveUint32())
            x_pos = self.socket.receiveFloat32()
            y_pos = self.socket.receiveFloat32()
            z_pos = self.socket.receiveFloat32()
            return [cmd, frame_number, x_pos, y_pos, z_pos]

        if cmd == CMD_PROCESS_IMAGE:
            frame_number = self.restore_frame_number(self.socket.receiveUint32())
            left_image = self.socket.receive()
            right_image = self.socket.receive()
            
            image_data = {
                "left_image": left_image,
                "right_image": right_image
            }
            return [cmd, frame_number, image_data]

        raise ValueError("Unsupported command received from socket: %s" % str(cmd))

    # Send a SoC command back to the connected MATLAB client
    def send_soc_command(self, cmd_array):
        command = list(cmd_array)
        cmd = int(command[0])

        self.socket.sendCmd(cmd)

        if cmd in COMMANDS_WITHOUT_ARGS:
            return

        if cmd in COMMANDS_WITH_ONE_SIGNED_INT:
            self.socket.sendInt32(command[1])
            return

        if cmd in COMMANDS_WITH_ONE_UINT:
            self.send_matlab_uint32(self.normalize_frame_number(command[1]))
            return

        if cmd == CMD_SEND_CALL:
            if command[1]:
                value = 1
            else:
                value = 0
            self.socket.sendUint8(value)
            if len(command) >= 3:
                self.send_matlab_uint32(self.normalize_frame_number(command[2]))
            return

        if cmd == CMD_LOG_DATA:
            self.send_matlab_uint32(self.normalize_frame_number(command[1]))
            self.send_matlab_float32(command[2])
            self.send_matlab_float32(command[3])
            self.send_matlab_float32(command[4])
            return

        if cmd == CMD_PROCESS_IMAGE:
            self.send_matlab_uint32(self.normalize_frame_number(command[1]))
            image_data = command[2]
            if isinstance(image_data, dict) and "left_image" in image_data and "right_image" in image_data:
                left_image = image_data["left_image"]
                right_image = image_data["right_image"]
            else:
                raise ValueError("processImage expects a stereo dict with left_image and right_image")
            self.socket.send(np.asarray(left_image, dtype=self.socket.image_dtype))
            self.socket.send(np.asarray(right_image, dtype=self.socket.image_dtype))
            return

        raise ValueError("Unsupported command for socket send: %s" % str(cmd))

    # MATLAB tcpclient read(..., "uint32") is currently consuming frame ids in host byte order.
    def send_matlab_uint32(self, value):
        self.socket.activeSocket().sendall(struct.pack("<I", int(value)))

    # MATLAB tcpclient read(..., "single") is currently consuming float values in host byte order.
    def send_matlab_float32(self, value):
        self.socket.activeSocket().sendall(struct.pack("<f", float(value)))

    # Convert frame numbers into the uint32 wire representation
    def normalize_frame_number(self, frame_number):
        if frame_number is None:
            return UNKNOWN_FRAME_NUMBER

        value = int(frame_number)
        if value < 0 or value >= UNKNOWN_FRAME_NUMBER:
            raise ValueError("frame_number must fit in uint32 and reserve 0xFFFFFFFF for unknown")
        return value

    # Convert the frame number back into Python side form
    def restore_frame_number(self, frame_number):
        if frame_number is None or int(frame_number) == UNKNOWN_FRAME_NUMBER:
            return None

        value = int(frame_number)
        # MATLAB tcpclient write(..., "uint32") on Windows is currently sending
        # host-endian bytes, while the Python protocol expects network order.
        # Normalize obviously byte-swapped frame numbers here so the rest of the
        # stack can stay unchanged until the GUI-side writer is cleaned up.
        if value > 0x00FFFFFF:
            swapped = struct.unpack("<I", struct.pack(">I", value))[0]
            if 0 <= swapped < value:
                return int(swapped)

        return value


# Define command line flags for the TCP adapter process
def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run the SoC side MATLAB TCP adapter"
    )
    parser.add_argument(
        "--host",
        default="",
        help="Interface to bind to, Defaults to all IPv4 interfaces"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9999,
        help="TCP port to listen on"
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=1920,
        help="Expected image width for processImage payloads"
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=1080,
        help="Expected image height for processImage payloads"
    )
    parser.add_argument(
        "--image-channels",
        type=int,
        default=3,
        help="Expected image channel count, Use 1 for grayscale and 3 for RGB"
    )
    parser.add_argument(
        "--overlay-bit",
        default=DEFAULT_OVERLAY_BIT,
        help="Optional PYNQ bitstream path. Leave empty to run without FPGA fast path"
    )
    parser.add_argument(
        "--fpga-ip-base",
        type=parse_int_auto,
        default=DEFAULT_FPGA_IP_BASE,
        help="AXI-Lite base address for the custom detector IP, accepts decimal or 0x hex"
    )
    parser.add_argument(
        "--fpga-ip-size",
        type=parse_int_auto,
        default=DEFAULT_FPGA_IP_SIZE,
        help="AXI-Lite register window size, accepts decimal or 0x hex"
    )
    parser.add_argument(
        "--fpga-timeout-s",
        type=float,
        default=15.0,
        help="Seconds to wait for the FPGA detector before timing out"
    )
    parser.add_argument(
        "--disable-hardware",
        "--disable-fpga-fast-path",
        dest="disable_hardware",
        action="store_true",
        help="Skip FPGA hardware and run the FAST algorithm in software during master mode"
    )
    parser.add_argument(
        "--disable-fast-path",
        action="store_true",
        help="Skip both hardware FAST and software FAST, and use the fallback algorithm directly in master mode"
    )
    return parser


def parse_int_auto(value):
    return int(str(value), 0)


def get_overlay_ip(overlay, ip_name):
    if not hasattr(overlay, ip_name):
        raise ValueError("IP '%s' was not found in the loaded overlay" % ip_name)
    return getattr(overlay, ip_name)


# Build a stereo DMA backed FPGA cache when overlay settings are provided
def build_fpga_cache(
    image_shape,
    overlay_bit="",
    ip_base=DEFAULT_FPGA_IP_BASE,
    ip_size=DEFAULT_FPGA_IP_SIZE,
    timeout_s=15.0,
):
    if not overlay_bit:
        return None

    left_dma_name = DEFAULT_LEFT_DMA_NAME
    right_dma_name = DEFAULT_RIGHT_DMA_NAME
    left_mask_dma_name = DEFAULT_LEFT_MASK_DMA_NAME
    right_mask_dma_name = DEFAULT_RIGHT_MASK_DMA_NAME
    from pynq import Overlay

    overlay = Overlay(overlay_bit)
    left_dma = get_overlay_ip(overlay, left_dma_name)
    right_dma = get_overlay_ip(overlay, right_dma_name)
    left_mask_dma = get_overlay_ip(overlay, left_mask_dma_name)
    right_mask_dma = get_overlay_ip(overlay, right_mask_dma_name)
    dma_engine = StereoDmaEngine(
        left_dma=left_dma,
        right_dma=right_dma,
        left_mask_dma=left_mask_dma,
        right_mask_dma=right_mask_dma
    )

    fpga_cache = PingPongFpgaCache(
        dma_engine=dma_engine,
        image_shape=image_shape,
        ip_base=ip_base,
        ip_size=ip_size,
        timeout_s=timeout_s
    )
    return overlay, fpga_cache


# Start the adapter from the command line and wait for a client
def main():
    args = build_argument_parser().parse_args()
    if args.host:
        host = args.host
    else:
        host = "0.0.0.0"
    print("Waiting for MATLAB client on %s:%d" % (host, args.port))
    
    if args.image_channels <= 1:
        image_shape = (args.image_height, args.image_width)
    else:
        image_shape = (args.image_height, args.image_width, args.image_channels)

    overlay = None
    fpga_cache = None
    if not args.disable_hardware:
        fpga_setup = build_fpga_cache(
            image_shape=image_shape,
            overlay_bit=args.overlay_bit,
            ip_base=args.fpga_ip_base,
            ip_size=args.fpga_ip_size,
            timeout_s=args.fpga_timeout_s
        )
        if fpga_setup is not None:
            overlay, fpga_cache = fpga_setup
            print("[FPGA] DMA backed PingPongFpgaCache configured")
    else:
        print("[FPGA] Hardware fast path disabled by flag")

    if not args.disable_hardware and fpga_cache is None:
        print("[FPGA] No DMA cache configured so hardware FAST is unavailable")

    adapter = MatlabServerAdapter(
        host=args.host,
        port=args.port,
        image_shape=image_shape,
        fpga_cache=fpga_cache,
        disable_hardware=args.disable_hardware,
        disable_fast_path=args.disable_fast_path,
    )
    stop_requested = {"value": False}

    def handle_stop_signal(signum, frame):
        del signum
        del frame
        stop_requested["value"] = True
        adapter.close()

    signal.signal(signal.SIGINT, handle_stop_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_stop_signal)

    try:
        adapter.serve_forever()
    except (ConnectionError, OSError) as exc:
        if stop_requested["value"]:
            print("TCP server stopped")
        else:
            print("TCP connection closed:", exc)
    except KeyboardInterrupt:
        print("TCP server stopped")


if __name__ == "__main__":
    main()
