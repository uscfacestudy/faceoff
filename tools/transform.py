import numpy as np


def rotate_around(angle, center, *points):
    """Rotate a set of points around a pivot."""

    matrix = np.zeros(shape=(2, 0))
    for x, y in points:
        matrix = np.append(matrix, np.matrix((x - center[0], y - center[1])).T, 1)
    rotation = np.matrix(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    matrix = rotation @ matrix
    for i in range(len(points)):
        x, y = matrix[:, i].flat
        yield x + center[0], y + center[1]
