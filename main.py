import os
import shutil
import math
import itertools
import json

import face_recognition
import PIL.Image
import PIL.ImageDraw
import numpy as np

from tools.landmarks import *


def get_stretched(directory, level):
    """Get all paths of faces with a certain stretch."""

    return filter(lambda n: n.endswith(str(level) + ".jpg"), os.listdir(directory))


def normalize_rotation(path):
    """Normalize the rotation of a face."""

    face_image = face_recognition.load_image_file(path)
    face_landmarks = face_recognition.face_landmarks(face_image)[0]
    angle = eye_angle(face_landmarks)
    face_landmarks["angle"] = angle

    with open("stimuli/data/" + os.path.basename(path) + ".json", "w") as file:
        json.dump(face_landmarks, file, indent=2)


def main():
    """Run with the script."""

    for path in get_stretched("stimuli/normal", 100):
        normalize_rotation("stimuli/normal/" + path)


if __name__ == "__main__":
    main()
