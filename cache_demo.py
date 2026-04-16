# -*- coding: utf-8 -*-

import os
import re
import sys
import zipfile

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared_protocol.soc_protocol import SoCProtocol, CMD_PROCESS_IMAGE, CMD_SLAVE_MODE, CMD_RESET


TEST_INPUT_DIR = os.path.join(os.path.dirname(__file__), "test_input")
LEFT_IMAGE_DIR = os.path.join(TEST_INPUT_DIR, "left_image")
RIGHT_IMAGE_DIR = os.path.join(TEST_INPUT_DIR, "right_image")
DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
ZIP_INPUT_PATH = os.path.join(DOWNLOADS_DIR, "Sample_Images-20260412T071646Z-3-001.zip")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# Return the trailing frame token from a filename, or the base name if no digits exist
def get_frame_token(filename):
    stem, _ = os.path.splitext(filename)
    match = re.search(r"(\d+)$", stem)
    if match is not None:
        return match.group(1)
    return stem


# Build a frame-token to path map for one stereo image folder
def build_image_map(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError("Missing image folder: %s" % folder_path)

    image_map = {}
    for entry in os.listdir(folder_path):
        file_path = os.path.join(folder_path, entry)
        if not os.path.isfile(file_path):
            continue

        _, extension = os.path.splitext(entry)
        if extension.lower() not in IMAGE_EXTENSIONS:
            continue

        frame_token = get_frame_token(entry)
        if frame_token in image_map:
            raise ValueError("Duplicate frame token '%s' in %s" % (frame_token, folder_path))
        image_map[frame_token] = file_path

    return image_map


# Build a frame-token to path map from the Blender-style filenames in Downloads
def build_downloads_image_map(prefix):
    image_map = {}
    if not os.path.isdir(DOWNLOADS_DIR):
        return image_map

    for entry in os.listdir(DOWNLOADS_DIR):
        if not entry.startswith(prefix):
            continue

        file_path = os.path.join(DOWNLOADS_DIR, entry)
        if not os.path.isfile(file_path):
            continue

        _, extension = os.path.splitext(entry)
        if extension.lower() not in IMAGE_EXTENSIONS:
            continue

        frame_token = get_frame_token(entry)
        if frame_token in image_map:
            raise ValueError("Duplicate frame token '%s' in %s" % (frame_token, DOWNLOADS_DIR))
        image_map[frame_token] = file_path

    return image_map


# Build a frame-token to zip-entry map from the sample stereo zip
def build_zip_image_map(prefix):
    image_map = {}
    if not os.path.isfile(ZIP_INPUT_PATH):
        return image_map

    with zipfile.ZipFile(ZIP_INPUT_PATH) as zip_file:
        for entry in zip_file.namelist():
            filename = os.path.basename(entry)
            if not filename.startswith(prefix):
                continue

            _, extension = os.path.splitext(filename)
            if extension.lower() not in IMAGE_EXTENSIONS:
                continue

            frame_token = get_frame_token(filename)
            if frame_token in image_map:
                raise ValueError("Duplicate frame token '%s' in %s" % (frame_token, ZIP_INPUT_PATH))
            image_map[frame_token] = entry

    return image_map


# Sort numeric frame tokens numerically and everything else lexically
def sort_frame_tokens(tokens):
    def sort_key(token):
        if token.isdigit():
            return (0, int(token))
        return (1, token)

    return sorted(tokens, key=sort_key)


# Load one image from disk and convert it from OpenCV BGR into RGB
def load_rgb_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image: %s" % file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Load one RGB image from a zip entry
def load_rgb_image_from_zip(entry_name):
    with zipfile.ZipFile(ZIP_INPUT_PATH) as zip_file:
        encoded_bytes = zip_file.read(entry_name)

    encoded_array = np.frombuffer(encoded_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image from zip entry: %s" % entry_name)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Run the fallback path in slave mode using stereo images from disk
def main():
    left_image_map = build_image_map(LEFT_IMAGE_DIR)
    right_image_map = build_image_map(RIGHT_IMAGE_DIR)
    load_image = load_rgb_image

    if not left_image_map and not right_image_map:
        left_image_map = build_downloads_image_map("blenderImgL")
        right_image_map = build_downloads_image_map("blenderImgR")
        load_image = load_rgb_image

    if not left_image_map and not right_image_map:
        left_image_map = build_zip_image_map("blenderImgL")
        right_image_map = build_zip_image_map("blenderImgR")
        load_image = load_rgb_image_from_zip

    shared_tokens = sort_frame_tokens(set(left_image_map.keys()) & set(right_image_map.keys()))
    if not shared_tokens:
        raise ValueError("No matching left/right stereo pairs were found in test_input, Downloads, or the sample zip")

    missing_left = sort_frame_tokens(set(right_image_map.keys()) - set(left_image_map.keys()))
    missing_right = sort_frame_tokens(set(left_image_map.keys()) - set(right_image_map.keys()))
    if missing_left:
        print("Ignoring right-only frames:", missing_left)
    if missing_right:
        print("Ignoring left-only frames:", missing_right)

    soc = SoCProtocol()
    print(soc.handle_incoming_command([CMD_SLAVE_MODE]))

    for frame_token in shared_tokens:
        left_path = left_image_map[frame_token]
        right_path = right_image_map[frame_token]
        left_image = load_image(left_path)
        right_image = load_image(right_path)
        pair_label = "%s %s" % (os.path.basename(left_path), os.path.basename(right_path))

        if frame_token.isdigit():
            frame_number = int(frame_token)
        else:
            frame_number = None

        result = soc.handle_incoming_command([
            CMD_PROCESS_IMAGE,
            frame_number,
            {
                "left_image": left_image,
                "right_image": right_image,
            }
        ])

        print("Pair %s:" % frame_token, pair_label)
        print(result)

    latest = soc.latest_frame.get()
    if latest is not None:
        print("Latest frame:", latest["frame_number"])

    print(soc.handle_incoming_command([CMD_RESET]))


if __name__ == "__main__":
    main()
