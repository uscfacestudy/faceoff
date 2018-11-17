import face_recognition
import PIL.Image
import PIL.ImageDraw
import numpy as np


LANDMARKS = {
    "chin": 17,
    "left_eyebrow": 5,
    "right_eyebrow": 5,
    "nose_bridge": 4,
    "nose_tip": 5,
    "left_eye": 6,
    "right_eye": 6,
    "top_lip": 12,
    "bottom_lip": 12}


def eye_angle(face_landmarks):
    """Compute the angle of the line between the eyes from horizontal."""

    left_eye = tuple(np.mean(face_landmarks["left_eye"], axis=0))
    right_eye = tuple(np.mean(face_landmarks["right_eye"], axis=0))
    raw_angle = np.arctan((left_eye[1] - right_eye[1]) / (right_eye[0] - left_eye[0]))
    return float(raw_angle)


def draw_landmarks(pil_image, face_landmarks):
    """Draw landmarks on a face."""

    drawing = PIL.ImageDraw.Draw(pil_image)
    for name in face_landmarks:
        drawing.line(face_landmarks[name])

    return pil_image


def landmarks_center(landmarks):
    """Provide a center point for a facial landmark set."""

    return landmarks["nose_bridge"][-1]


def landmarks_size(landmarks):
    """Find with width and height bounding the face."""

    left = right = top = bottom = 0
    for name in LANDMARKS:
        for x, y in landmarks[name]:
            left = min(left, x)
            right = max(right, x)
            bottom = min(bottom, y)
            top = max(top, y)
    return right - left, abs(top - bottom)  # can't remember which way the axis is
