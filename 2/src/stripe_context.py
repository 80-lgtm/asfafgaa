import numpy as np


class StripeContextScorer:
    """Estimate how strongly a patch resembles the image's dominant stripe direction."""

    def __init__(self, strength: float = 1.0):
        self.strength = float(strength)

    def consistency(self, patch: np.ndarray) -> float:
        patch = np.asarray(patch, dtype=np.float32)
        if patch.ndim != 2 or min(patch.shape) < 2:
            raise ValueError("patch must be a two-dimensional array of at least 2x2 pixels")
        variance_x = float(np.mean(np.diff(patch, axis=0) ** 2))
        variance_y = float(np.mean(np.diff(patch, axis=1) ** 2))
        total = variance_x + variance_y
        if total == 0:
            return 0.0
        dominance = max(variance_x, variance_y) / total
        return float(np.clip((dominance - 0.5) * 2 * self.strength, 0.0, 1.0))


def global_stripe_strength(image: np.ndarray) -> float:
    """Measure directional structure; uniform images receive exactly zero."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError("image must be a two-dimensional array of at least 2x2 pixels")
    dx = float(np.mean(np.diff(image, axis=1) ** 2))
    dy = float(np.mean(np.diff(image, axis=0) ** 2))
    total = dx + dy
    if total <= np.finfo(np.float32).eps:
        return 0.0
    return float(abs(dx - dy) / total)


def fft_stripe_strength(image: np.ndarray) -> float:
    """Spectral concentration of a centered image; stripes give a dominant peak."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError("image must be a two-dimensional array of at least 2x2 pixels")
    centered = image - float(image.mean())
    if not np.any(centered):
        return 0.0
    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(centered)))
    cy, cx = magnitude.shape[0] // 2, magnitude.shape[1] // 2
    magnitude[cy, cx] = 0.0
    energy = float(np.linalg.norm(magnitude))
    if energy <= np.finfo(np.float32).eps:
        return 0.0
    return float(np.clip(float(magnitude.max()) / energy, 0.0, 1.0))


def frequency_consistency_map(image: np.ndarray, *, grid_shape) -> np.ndarray:
    """Local directional agreement scaled by frequency-domain stripe strength."""
    image = np.asarray(image, dtype=np.float32)
    rows, cols = grid_shape
    if rows <= 0 or cols <= 0:
        raise ValueError("grid dimensions must be positive")
    strength = fft_stripe_strength(image)
    if strength == 0.0:
        return np.zeros((rows, cols), dtype=np.float32)
    global_dx = float(np.mean(np.diff(image, axis=1) ** 2))
    global_dy = float(np.mean(np.diff(image, axis=0) ** 2))
    dominant_axis = 1 if global_dx >= global_dy else 0
    scorer = StripeContextScorer()
    result = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            y0, y1 = row * image.shape[0] // rows, (row + 1) * image.shape[0] // rows
            x0, x1 = col * image.shape[1] // cols, (col + 1) * image.shape[1] // cols
            patch = image[y0:y1, x0:x1]
            local_dx = float(np.mean(np.diff(patch, axis=1) ** 2))
            local_dy = float(np.mean(np.diff(patch, axis=0) ** 2))
            if local_dx + local_dy == 0:
                continue
            local_axis = 1 if local_dx >= local_dy else 0
            if local_axis == dominant_axis:
                result[row, col] = strength * scorer.consistency(patch)
    return result


def stripe_consistency_map(image: np.ndarray, *, grid_shape) -> np.ndarray:
    """Combine global stripe evidence with local directional agreement."""
    image = np.asarray(image, dtype=np.float32)
    rows, cols = grid_shape
    if rows <= 0 or cols <= 0:
        raise ValueError("grid dimensions must be positive")
    strength = global_stripe_strength(image)
    if strength == 0.0:
        return np.zeros((rows, cols), dtype=np.float32)
    global_dx = float(np.mean(np.diff(image, axis=1) ** 2))
    global_dy = float(np.mean(np.diff(image, axis=0) ** 2))
    dominant_axis = 1 if global_dx >= global_dy else 0
    scorer = StripeContextScorer()
    result = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            y0, y1 = row * image.shape[0] // rows, (row + 1) * image.shape[0] // rows
            x0, x1 = col * image.shape[1] // cols, (col + 1) * image.shape[1] // cols
            patch = image[y0:y1, x0:x1]
            local_dx = float(np.mean(np.diff(patch, axis=1) ** 2))
            local_dy = float(np.mean(np.diff(patch, axis=0) ** 2))
            if local_dx + local_dy == 0:
                continue
            local_axis = 1 if local_dx >= local_dy else 0
            if local_axis == dominant_axis:
                result[row, col] = strength * scorer.consistency(patch)
    return result
