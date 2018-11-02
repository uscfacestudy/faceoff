import math

import PIL.Image


def d(a, b=None):
    """Euclidean distance between two points or the length of one vector."""

    if b is not None:
        a = (b[0] - a[0], b[1] - a[1])
    return math.sqrt(a[0] * a[0] + a[1] * a[1])


def clamp(value, floor, ceiling):
    """Clamp a value."""

    return min(max(value, floor), ceiling)


def warp_image(image, sources, destinations):
    """Warp an image via a set of point transforms."""

    result = PIL.Image.new("L", image.size, "black")
    image_pixels = image.load()
    result_pixels = result.load()

    for x in range(image.size[1]):
        for y in range(image.size[0]):

            dx = dy = 0
            for source, destination in zip(sources, destinations):
                delta = destination[0] - source[0], destination[1] - source[1]
                magnitude = 1 / (3 * (d((x, y), destination) / d(delta)) ** 4 + 1)
                dx += magnitude * delta[0]
                dy += magnitude * delta[1]
            nx = clamp(x + math.floor(dx), 0, image.size[0] - 1)
            ny = clamp(y + math.floor(dy), 0, image.size[1] - 1)
            result_pixels[x, y] = image_pixels[nx, ny]

    return result
