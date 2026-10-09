from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read_nifti(path: str | Path, pixel_type: int = sitk.sitkFloat32) -> tuple[np.ndarray, sitk.Image]:
    image = sitk.ReadImage(str(path), pixel_type)
    array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    return array, image


def write_nifti(array: np.ndarray, reference: sitk.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array.astype(np.float32, copy=False))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path))
