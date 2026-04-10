# -*- coding: utf-8 -*-
"""
SoC side TCP server for MATLAB/Python integration
Incoming commands are expected to use the binary layouts in numpysocket.py and
carry the command array protocol defined in soc_protocol.py which was outlined by August.
"""

import argparse
import signal

import numpy as np

from fpga_buffer_manager import PingPongFpgaCache
from numpysocket import NumpySocket
from soc_protocol import (
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

OVERLAY_BIT = ""
DMA_NAME = ""


COMMANDS_WITHOUT_ARGS = {
    CMD_REQUEST_LATEST_IMAGE,
    CMD_RESET,
    CMD_SLAVE_MODE,
    CMD_SLAVE_MODE_READY,
    CMD_STOP_CAPTURE,
}

COMMANDS_WITH_ONE_INT = {
    CMD_REQUEST_IMAGE_AT_FRAME,
    CMD_REQUEST_NTH_NEXT_IMAGE,
    CMD_REQUEST_NTH_PREVIOUS_IMAGE,
}


class MatlabServerAdapter(object):
    # Set up the TCP server and bind it to a SoC protocol instance
    def __init__(self, host="", port=9999, image_shape=None, protocol=None, fpga_cache=None):
        self.host = host
        self.port = port
        if image_shape is None:
            image_shape = (1080, 1920, 3)
        self.image_shape = tuple(image_shape)
        self.socket = NumpySocket(image_shape=self.image_shape)
        self.fpga_cache = fpga_cache
        self.protocol = protocol or SoCProtocol(
            command_sender=self.send_soc_command,
            fpga_cache=self.fpga_cache
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

        if cmd in COMMANDS_WITH_ONE_INT:
            return [cmd, self.socket.receiveInt32()]

        if cmd == CMD_SEND_CALL:
            return [cmd, self.socket.receiveUint8()]

        if cmd == CMD_LOG_DATA:
            frame_number = self.restore_frame_number(self.socket.receiveInt32())
            x_pos = self.socket.receiveFloat32()
            y_pos = self.socket.receiveFloat32()
            z_pos = self.socket.receiveFloat32()
            return [cmd, frame_number, x_pos, y_pos, z_pos]

        if cmd == CMD_PROCESS_IMAGE:
            frame_number = self.restore_frame_number(self.socket.receiveInt32())
            image_data = self.socket.receive()
            return [cmd, frame_number, image_data]

        raise ValueError("Unsupported command received from socket: %s" % str(cmd))

    # Send a SoC command back to the connected MATLAB client
    def send_soc_command(self, cmd_array):
        command = list(cmd_array)
        cmd = int(command[0])

        self.socket.sendCmd(cmd)

        if cmd in COMMANDS_WITHOUT_ARGS:
            return

        if cmd in COMMANDS_WITH_ONE_INT:
            self.socket.sendInt32(command[1])
            return

        if cmd == CMD_SEND_CALL:
            if command[1]:
                value = 1
            else:
                value = 0
            self.socket.sendUint8(value)
            return

        if cmd == CMD_LOG_DATA:
            self.socket.sendInt32(self.normalize_frame_number(command[1]))
            self.socket.sendFloat32(command[2])
            self.socket.sendFloat32(command[3])
            self.socket.sendFloat32(command[4])
            return

        if cmd == CMD_PROCESS_IMAGE:
            self.socket.sendInt32(self.normalize_frame_number(command[1]))
            self.socket.send(np.asarray(command[2], dtype=self.socket.image_dtype))
            return

        raise ValueError("Unsupported command for socket send: %s" % str(cmd))

    # Convert frame numbers into the int32 wire representation
    def normalize_frame_number(self, frame_number):
        if frame_number is None:
            return -1
        return int(frame_number)

    # Convert the frame number back into Python side form
    def restore_frame_number(self, frame_number):
        if frame_number is None or frame_number < 0:
            return None
        return frame_number


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
    return parser


# Build a DMA-backed FPGA cache when overlay settings are provided
def build_fpga_cache(image_shape):
    if not OVERLAY_BIT and not DMA_NAME:
        return None

    if not OVERLAY_BIT or not DMA_NAME:
        raise ValueError("Both OVERLAY_BIT and DMA_NAME must be set for fpga_cache setup")

    from pynq import Overlay

    overlay = Overlay(OVERLAY_BIT)

    if not hasattr(overlay, DMA_NAME):
        raise ValueError("DMA IP '%s' was not found in the loaded overlay" % DMA_NAME)

    dma_engine = getattr(overlay, DMA_NAME)
    fpga_cache = PingPongFpgaCache(
        dma_engine=dma_engine,
        image_shape=image_shape
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
    fpga_setup = build_fpga_cache(image_shape)
    if fpga_setup is not None:
        overlay, fpga_cache = fpga_setup

    adapter = MatlabServerAdapter(
        host=args.host,
        port=args.port,
        image_shape=image_shape,
        fpga_cache=fpga_cache
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
    except (ConnectionError, OSError):
        if stop_requested["value"]:
            print("TCP server stopped")
        else:
            print("TCP connection closed")
    except KeyboardInterrupt:
        print("TCP server stopped")


if __name__ == "__main__":
    main()
