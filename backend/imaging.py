"""
Pipeline stages 1-2: image loading and preprocessing.

Turns an uploaded image, or a page rasterized from a scanned PDF, into an
ordered list of single-line crops ready for handwriting recognition.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MIN_LINE_HEIGHT = 8
MAX_LINE_HEIGHT = 400
LINE_PAD = 6
MERGE_GAP = 4
INK_RATIO = 0.015
MAX_SKEW_ANGLE = 15.0


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path.name}")
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(
        gray, None, h=10, templateWindowSize=7, searchWindowSize=21
    )


def binarize(gray: np.ndarray) -> np.ndarray:
    """Inverted adaptive threshold, so ink is white on black."""
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )


def skew_angle(mask: np.ndarray) -> float:
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 50:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90

    return 0.0 if abs(angle) > MAX_SKEW_ANGLE else float(angle)


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    angle = skew_angle(binarize(gray))
    if not angle:
        return gray, 0.0

    height, width = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True ranges in a boolean array."""
    if not flags.any():
        return []

    changes = np.flatnonzero(np.diff(flags.astype(np.int8))) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(flags)]))
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if flags[start]
    ]


def line_bands(mask: np.ndarray) -> list[tuple[int, int]]:
    """Row ranges holding ink, hairline noise dropped and split strokes merged."""
    ink_per_row = np.count_nonzero(mask, axis=1)
    threshold = max(1, int(mask.shape[1] * INK_RATIO))
    bands = _runs(ink_per_row >= threshold)

    merged: list[tuple[int, int]] = []
    for start, end in bands:
        if merged and start - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    return [
        (start, end)
        for start, end in merged
        if MIN_LINE_HEIGHT <= end - start <= MAX_LINE_HEIGHT
    ]


def segment_lines(gray: np.ndarray) -> list[np.ndarray]:
    """
    Crop one image per handwritten line.

    Boundaries come from the binarized mask, but crops are taken from the
    grayscale image: thresholding discards the stroke weight and grey levels
    that TrOCR relies on.
    """
    bands = line_bands(binarize(gray))
    height, width = gray.shape[:2]

    if not bands:
        return [gray] if gray.size else []

    return [
        gray[max(0, start - LINE_PAD):min(height, end + LINE_PAD), 0:width]
        for start, end in bands
    ]


def preprocess_image(image: np.ndarray) -> tuple[list[np.ndarray], float]:
    gray = denoise(to_gray(image))
    gray, angle = deskew(gray)
    return segment_lines(gray), angle


def preprocess_file(path: Path) -> tuple[list[np.ndarray], float]:
    return preprocess_image(load_image(path))
