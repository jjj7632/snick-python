import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared_protocol.numpysocket import NumpySocket
from shared_protocol.image_cache import create_dummy_image
from shared_protocol.soc_protocol import CMD_LOG_DATA, CMD_PROCESS_IMAGE, CMD_REQUEST_LATEST_IMAGE

BOARD_IP = "10.0.110.2"
PORT = 9999
IMAGE_SHAPE = (1080, 1920, 3)

sock = NumpySocket(image_shape=IMAGE_SHAPE)
sock.start_client(BOARD_IP, PORT)

left_image = create_dummy_image(IMAGE_SHAPE[1], IMAGE_SHAPE[0], 120, channels=IMAGE_SHAPE[2])
right_image = create_dummy_image(IMAGE_SHAPE[1], IMAGE_SHAPE[0], 125, channels=IMAGE_SHAPE[2])
cmd = sock.receiveCmd()
if cmd == CMD_REQUEST_LATEST_IMAGE:
    sock.sendCmd(CMD_PROCESS_IMAGE)
    sock.sendInt32(42)
    sock.send(left_image)
    sock.send(right_image)
else:
    print("Received unexpected command:", cmd)

cmd = sock.receiveCmd()
if cmd == CMD_LOG_DATA:
    frame_number = sock.receiveInt32()
    x_pos = sock.receiveFloat32()
    y_pos = sock.receiveFloat32()
    z_pos = sock.receiveFloat32()
    print([cmd, frame_number, x_pos, y_pos, z_pos])
else:
    print("Received unexpected command:", cmd)

sock.close()
