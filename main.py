import os
import sys
import shutil
import math
import itertools
import json

from typing import List, Dict, Any

import face_recognition
import PIL.Image
import PIL.ImageDraw
import numpy as np

from tools.landmarks import *
from tools.transform import *

from cache import Cache

cache = Cache()


LANDMARKS = (
    "chin", "left_eyebrow", "right_eyebrow",
    "nose_bridge", "nose_tip", "left_eye", "right_eye",
    "top_lip", "bottom_lip")


def get_stretched(directory, level):
    """Get all paths of faces with a certain stretch."""

    return filter(lambda n: n.endswith(str(level) + ".jpg"), os.listdir(directory))


@cache(persist=True,
       serialize=lambda path: path + "_landmarks",  # serialize from the path
       file=lambda path: os.path.basename(path) + ".json",
       store=json.dump,
       retrieve=json.load)
def extract_landmark_data(path: str) -> Dict[str, Any]:
    """Normalize the rotation of a face."""

    face_image = face_recognition.load_image_file(path)
    face_landmarks = face_recognition.face_landmarks(face_image)[0]
    angle = eye_angle(face_landmarks)
    return {"landmarks": face_landmarks, "angle": angle}


def normalize_landmarks(data: Dict[str, Any]):
    """Center around the middle of the face and rotate."""

    for name in LANDMARKS:
        data["landmarks"][name] = tuple(rotate_around(data["angle"], (0, 0), *data["landmarks"][name]))
    return data


def find_center(landmarks):
    """Provide a center point for a facial landmark set."""

    return landmarks["nose_bridge"][-1]


def average_landmarks(landmarks=None, *, of: List[Dict[str, Any]]=[]):
    """This is a needlessly complicated way to compute an average.

    It's more a proof of concept, to be honest. We're abusing the fact
    that mutable default parameters retain their value between calls.
    """

    if landmarks is not None:
        of.append(landmarks)
        return

    average = {}
    for name in LANDMARKS:
        average[name] = []
        for i in range(len(of[0]["landmarks"][name])):
            x = y = 0
            count = len(of)
            for j in range(count):
                center = find_center(of[j]["landmarks"])
                x += of[j]["landmarks"][name][i][0] - center[0]
                y += of[j]["landmarks"][name][i][1] - center[1]
            average[name].append((x/count, y/count))

    return average


def main():
    """Run with the script."""

    for path in get_stretched("stimuli/normal", 100):
        landmarks = extract_landmark_data("stimuli/normal/" + path)
        landmarks = normalize_landmarks(landmarks)
        average_landmarks(landmarks)

    average = average_landmarks()


    # for path in os.listdir("stimuli/data/"):
    #     if path.endswith("json"):
    #         print(normalize_landmarks("stimuli/data/" + path))
    #         break

    center = find_center(extract_landmark_data("stimuli/normal/c01_100.jpg")["landmarks"])
    draw_landmarks("stimuli/normal/c01_100.jpg", custom_landmarks=average, custom_center=center).show()


if __name__ == "__main__":
    main()
