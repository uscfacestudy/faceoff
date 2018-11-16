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


def draw_landmarks(path, custom_landmarks=None, custom_center=(0, 0)):
    """Draw landmarks on a face."""

    face_image = face_recognition.load_image_file(path)
    pil_image = PIL.Image.fromarray(face_image)

    # face_location = face_recognition.face_locations(face_image)[0]
    face_landmarks = custom_landmarks or face_recognition.face_landmarks(face_image)[0]

    drawing = PIL.ImageDraw.Draw(pil_image)
    # top, right, bottom, left = face_location
    # drawing.rectangle(((left, top), (right, bottom)))
    for name in face_landmarks:
        drawing.line(list((x + custom_center[0], y + custom_center[1]) for x, y in face_landmarks[name]))

    return pil_image
