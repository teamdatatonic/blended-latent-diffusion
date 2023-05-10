from PIL import Image
from scipy.ndimage import binary_dilation

import numpy as np


def resize_image(
    image: np.ndarray,
    dest_width: int,
    color_type: str,
    resample_method: Image.Resampling,
) -> np.ndarray:
    image_dimensions = image.shape
    dest_dimensions = (
        int(np.ceil(image_dimensions[1] * (dest_width / image_dimensions[0]))),
        dest_width,
    )
    image = Image.fromarray(image).convert(color_type)
    image = image.resize(dest_dimensions, resample_method)
    return np.array(image)


def dilate_mask(mask: Image.Image, dilation_iterations: int) -> np.ndarray:
    masks_array = []
    for i in reversed(range(dilation_iterations)):
        k_size = 3 + 2 * i
        masks_array.append(binary_dilation(mask, structure=np.ones((k_size, k_size))))
    masks_array.append(mask)
    masks_array = np.array(masks_array).astype(np.float32)
    masks_array = masks_array[:, np.newaxis, :]
    return masks_array


def make_bitmask(mask_image: Image.Image, dtype: np.dtype) -> np.ndarray:
    mask_array = (np.array(mask_image).astype(np.float32) / 255.0)[None, None]
    mask_array = np.where(mask_array < 0.5, 0, 1)
    return mask_array.astype(dtype)
