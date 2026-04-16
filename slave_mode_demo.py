import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared_protocol.numpysocket import NumpySocket
from shared_protocol.image_cache import create_dummy_image
from shared_protocol.soc_protocol import (
    CMD_LOG_DATA,
    CMD_PROCESS_IMAGE,
    CMD_REQUEST_LATEST_IMAGE,
    CMD_RESET,
    CMD_SLAVE_MODE,
    CMD_SLAVE_MODE_READY,
)

BOARD_IP = "10.0.110.2"
PORT = 9999
IMAGE_SHAPE = (1080, 1920, 3)


# Read and print one logData command from the SoC.
def read_log_data(sock):
    cmd = sock.receiveCmd()
    if cmd != CMD_LOG_DATA:
        raise ValueError("Expected logData command, received %s" % str(cmd))

    frame_number = sock.receiveInt32()
    x_pos = sock.receiveFloat32()
    y_pos = sock.receiveFloat32()
    z_pos = sock.receiveFloat32()
    print([cmd, frame_number, x_pos, y_pos, z_pos])


# Drive the SoC through slave mode, one replay frame, and reset.
def main():
    sock = NumpySocket(image_shape=IMAGE_SHAPE)
    sock.start_client(BOARD_IP, PORT)

    first_cmd = sock.receiveCmd()
    if first_cmd == CMD_REQUEST_LATEST_IMAGE:
        print("Ignoring initial master-mode image request")
    else:
        raise ValueError("Expected initial requestLatestImage, received %s" % str(first_cmd))

    sock.sendCmd(CMD_SLAVE_MODE)

    ready_cmd = sock.receiveCmd()
    if ready_cmd != CMD_SLAVE_MODE_READY:
        raise ValueError("Expected slaveModeReady, received %s" % str(ready_cmd))
    print([ready_cmd])

    left_image = create_dummy_image(IMAGE_SHAPE[1], IMAGE_SHAPE[0], 180, channels=IMAGE_SHAPE[2])
    right_image = create_dummy_image(IMAGE_SHAPE[1], IMAGE_SHAPE[0], 185, channels=IMAGE_SHAPE[2])
    sock.sendCmd(CMD_PROCESS_IMAGE)
    sock.sendInt32(200)
    sock.send(left_image)
    sock.send(right_image)

    read_log_data(sock)

    sock.sendCmd(CMD_RESET)

    next_cmd = sock.receiveCmd()
    if next_cmd != CMD_REQUEST_LATEST_IMAGE:
        raise ValueError("Expected requestLatestImage after reset, received %s" % str(next_cmd))
    print([next_cmd])

    sock.close()


if __name__ == "__main__":
    main()
