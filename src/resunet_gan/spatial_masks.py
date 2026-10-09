from __future__ import annotations

import numpy as np
from scipy import ndimage


def physical_perilesional_bands(
    roi: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    inner_radius_mm: float = 5.0,
    outer_radius_mm: float | None = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a 0-inner ring and, when requested, an inner-outer ring."""
    if roi.ndim not in (3, 4):
        raise ValueError(f"ROI must have shape [D,H,W] or [C,D,H,W], got {roi.shape}")
    if inner_radius_mm <= 0:
        raise ValueError("Context radius must be positive")
    if outer_radius_mm is not None and outer_radius_mm <= inner_radius_mm:
        raise ValueError("Context radii must satisfy 0 < inner_radius_mm < outer_radius_mm")
    roi_channels = roi[None] if roi.ndim == 3 else roi
    spacing_x, spacing_y, spacing_z = (float(value) for value in spacing_xyz)
    sampling_zyx = (spacing_z, spacing_y, spacing_x)
    near = np.zeros_like(roi_channels, dtype=np.float32)
    far = np.zeros_like(roi_channels, dtype=np.float32)
    for channel, mask in enumerate(roi_channels > 0):
        if not np.any(mask):
            continue
        distance = ndimage.distance_transform_edt(~mask, sampling=sampling_zyx)
        near[channel] = ((distance > 0.0) & (distance <= inner_radius_mm)).astype(np.float32)
        if outer_radius_mm is not None:
            far[channel] = ((distance > inner_radius_mm) & (distance <= outer_radius_mm)).astype(np.float32)
    if roi.ndim == 3:
        return near[0], far[0]
    return near, far
