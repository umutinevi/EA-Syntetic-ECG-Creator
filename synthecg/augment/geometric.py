import cv2
import numpy as np


def apply_rotation_array(img: np.ndarray, rng: np.random.Generator, max_degrees: float = 1.5) -> np.ndarray:
    """Apply a small in-plane rotation."""
    angle = rng.uniform(-max_degrees, max_degrees)
    rows, cols = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE)


def apply_jpeg_compression(img: np.ndarray, rng: np.random.Generator, quality_range: tuple[int, int] = (72, 92)) -> np.ndarray:
    """Round-trip through JPEG compression to simulate scan/photo artifacts."""
    quality = int(rng.integers(quality_range[0], quality_range[1] + 1))
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else img


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
