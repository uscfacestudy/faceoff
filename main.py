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
from tools.warp import *

from cache import Cache

cache = Cache()


def get_stretched(directory, level):
    """Get all paths of faces with a certain stretch."""

    return filter(lambda n: n.endswith(str(level) + ".jpg"), os.listdir(directory))


@cache(persist=True,
       serialize=lambda path: path + "_landmarks",  # serialize from the path
       file=lambda path: os.path.basename(path) + ".json",
       store=json.dump,
       retrieve=lambda file: force_tuples(json.load(file)))
def extract_landmark_data(path: str) -> Dict[str, Any]:
    """Normalize the rotation of a face."""

    face_image = face_recognition.load_image_file(path)
    face_landmarks = face_recognition.face_landmarks(face_image)[0]
    angle = eye_angle(face_landmarks)
    center = landmarks_center(face_landmarks)
    size = landmarks_size(face_landmarks)
    return {"landmarks": face_landmarks, "angle": angle, "center": center, "size": size}


def normalize_landmarks(data: Dict[str, Any]):
    """Center around the middle of the face and rotate."""

    for name in LANDMARKS:
        data["landmarks"][name] = tuple(rotate_around(data["angle"], (0, 0), *data["landmarks"][name]))
    return data


@cache(persist=True,
       file="average.json",
       store=json.dump,
       retrieve=lambda file: force_tuples(json.load(file)))
def average_landmarks():
    """Compute the average for all normals."""

    of = []
    for path in get_stretched("stimuli/normal", 100):
        landmarks = extract_landmark_data("stimuli/normal/" + path)
        of.append(normalize_landmarks(landmarks))

    average = {}
    for name, features in LANDMARKS.items():
        average[name] = []
        for i in range(features):
            x = y = 0
            count = len(of)
            for j in range(count):
                center = of[j]["center"]
                width, height = of[j]["size"]
                x += (of[j]["landmarks"][name][i][0] - center[0]) / height
                y += (of[j]["landmarks"][name][i][1] - center[1]) / height
            average[name].append((x/count, y/count))

    return average


def scale_landmarks(landmarks, center, size):
    """Create a scaled and centered copy of landmarks."""

    copy = {}
    for name in LANDMARKS:
        copy[name] = []
        for x, y in landmarks[name]:
            copy[name].append((x * size[0] + center[0], y * size[1] + center[1]))
    return copy


def force_tuples(landmarks):
    """Convert lists to tuples."""

    for name in LANDMARKS:
        landmarks[name] = list(map(tuple, landmarks[name]))
    return landmarks


def main():
    """Run with the script."""

    average = average_landmarks()

    path = "stimuli/normal/c01_100.jpg"
    face_image = face_recognition.load_image_file(path)
    pil_image = PIL.Image.fromarray(face_image)

    example = extract_landmark_data(path)
    scaled_average = scale_landmarks(average, example["center"], example["size"])

    draw_landmarks(pil_image, scaled_average)
    # pil_image.show()

    sources = []
    destinations = []
    for name, features in LANDMARKS.items():
        for i in range(features):
            sources.append(example["landmarks"][name][i])
            destinations.append(tuple(map(int, scaled_average[name][i])))

    result = warp_image(pil_image, sources, destinations)

    draw_landmarks(result, example["landmarks"])

    result.show()


if __name__ == "__main__":
    main()
