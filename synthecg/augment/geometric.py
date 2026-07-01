import cv2
import numpy as np


def apply_perspective_warp_array(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a mild perspective warp to simulate scanner angle variation."""
    rows, cols = img.shape[:2]
    max_shift = int(min(rows, cols) * 0.015)

    src = np.float32([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]])
    dst = np.float32(
        [
            [rng.integers(0, max_shift + 1), rng.integers(0, max_shift + 1)],
            [cols - 1 - rng.integers(0, max_shift + 1), rng.integers(0, max_shift + 1)],
            [cols - 1 - rng.integers(0, max_shift + 1), rows - 1 - rng.integers(0, max_shift + 1)],
            [rng.integers(0, max_shift + 1), rows - 1 - rng.integers(0, max_shift + 1)],
        ]
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE)


def apply_perspective_warp(image_path: str, output_path: str, seed: int | None = None) -> None:
    """Apply perspective warp to an image file."""
    rng = np.random.default_rng(seed)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    cv2.imwrite(output_path, apply_perspective_warp_array(img, rng))
