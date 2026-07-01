import cv2
import numpy as np

from synthecg.config import AugmentConfig


def apply_paper_artifacts(image_path: str, output_path: str, config: AugmentConfig | None = None) -> list[str]:
    """Apply scan-style paper artifacts: tint, noise, lighting gradient, and blur."""
    config = config or AugmentConfig()
    rng = np.random.default_rng(config.seed)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    applied: list[str] = []

    if config.profile == "clean":
        cv2.imwrite(output_path, img)
        return applied

    tint = np.array([214, 214, 243], dtype=np.float32) / 255.0
    img = np.clip(img * tint, 0, 255).astype(np.uint8)
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
    img = np.clip(img * lighting_variation, 0, 255).astype(np.uint8)
    applied.append("lighting_gradient")

    img = cv2.GaussianBlur(img, (3, 3), 0)
    applied.append("gaussian_blur")

    if config.profile == "clinical" and rng.random() < 0.5:
        from synthecg.augment.geometric import apply_perspective_warp_array

        img = apply_perspective_warp_array(img, rng)
        applied.append("perspective_warp")

    cv2.imwrite(output_path, img)
    return applied
