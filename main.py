import os

import face_recognition
import PIL.Image
import PIL.ImageDraw
import numpy as np


def eye_angle(face_landmarks):
    """Compute the angle of the line between the eyes from horizontal."""

    left_eye = tuple(np.mean(face_landmarks["left_eye"], axis=0))
    right_eye = tuple(np.mean(face_landmarks["right_eye"], axis=0))
    raw_angle = np.arctan((left_eye[1] - right_eye[1]) / (right_eye[0] - left_eye[0]))
    return float(raw_angle)


def nose_angle(face_landmarks):
    """Compute the angle from vertical."""


def rotate_around(angle, center, *pairs):
    """Rotate a set of points around a pivot."""

    points = np.zeros(shape=(2, 0))
    for x, y in pairs:
        points = np.append(points, np.matrix((x - center[0], y - center[1])).T, 1)
    rotation = np.matrix(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    points = rotation @ points
    for i in range(len(pairs)):
        x, y = points[:, i].flat
        yield x + center[0], y + center[1]


def normalize_face(path):
    """Normalize the rotation of a face."""

    face_image = face_recognition.load_image_file(path)
    pil_image = PIL.Image.fromarray(face_image)

    face_location = face_recognition.face_locations(face_image)[0]
    face_landmarks = face_recognition.face_landmarks(face_image)[0]

    angle = eye_angle(face_landmarks)

    top, right, bottom, left = face_location
    center = (left + right) / 2, (top + bottom) / 2
    corners = rotate_around(-angle, center, (left, top), (right, bottom))

    pil_image = pil_image.rotate(-np.rad2deg(angle))

    drawing = PIL.ImageDraw.Draw(pil_image)
    drawing.rectangle(tuple(corners))

    return pil_image




#drawing = PIL.ImageDraw.Draw(pil_image)
#center = (right + left) / 2, (top + bottom) / 2
#for name in face_landmarks:
#    drawing.line(face_landmarks[name])



def main():
    """Run with the script."""

    rotated = normalize_face("stimuli/c01_100.jpg")
    rotated.show()


if __name__ == "__main__":
    main()
