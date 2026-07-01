import cv2
import numpy as np

from synthecg.config import AugmentConfig
from synthecg.augment.geometric import apply_perspective_warp_array


def apply_paper_artifacts_to_arrays(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    config: AugmentConfig | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Apply scan-style paper artifacts to image and optional mask arrays."""
    config = config or AugmentConfig()
    rng = np.random.default_rng(config.seed)
    applied: list[str] = []

    if config.profile == "clean":
        return img, mask, applied

    tint = np.array([214, 214, 243], dtype=np.float32) / 255.0
    img = np.clip(img.astype(np.float32) * tint, 0, 255).astype(np.uint8)
    applied.append("pink_tint")

    noise = rng.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    applied.append("gaussian_noise")

    rows, cols, _ = img.shape
    gradient = np.zeros((rows, cols), dtype=np.float32)
    for i in range(cols):
        gradient[:, i] = i / cols
    gradient = np.stack([gradient] * 3, axis=-1)
    lighting_variation = 0.90 + 0.10 * gradient
    img = np.clip(img.astype(np.float32) * lighting_variation, 0, 255).astype(np.uint8)
    applied.append("lighting_gradient")

    img = cv2.GaussianBlur(img, (3, 3), 0)
    applied.append("gaussian_blur")

    if config.profile == "clinical" and rng.random() < 0.5:
        img = apply_perspective_warp_array(img, rng)
        if mask is not None:
            mask = apply_perspective_warp_array(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), rng)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        applied.append("perspective_warp")

    return img, mask, applied


def apply_paper_artifacts(
    image_path: str,
    output_path: str,
    config: AugmentConfig | None = None,
    mask_path: str | None = None,
    mask_output_path: str | None = None,
) -> list[str]:
    """Apply scan-style paper artifacts to image files on disk."""
    config = config or AugmentConfig()

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
    img, mask, applied = apply_paper_artifacts_to_arrays(img, mask, config)

    cv2.imwrite(output_path, img)
    if mask is not None and mask_output_path:
        cv2.imwrite(mask_output_path, mask)

    return applied
