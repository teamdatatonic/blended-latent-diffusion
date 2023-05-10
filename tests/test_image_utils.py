from ldm.utils.image_utils import resize_image, dilate_mask, make_bitmask
from PIL import Image

import numpy as np


def test_resize_image(resource_path):
    input_image = np.array(Image.open(resource_path / "input.png"))
    expected_image = np.array(Image.open(resource_path / "expected.png"))
    resized_image = resize_image(
        input_image, dest_width=35, color_type="RGB", resample_method=Image.LANCZOS
    )

    assert np.array_equiv(resized_image, expected_image)


def test_dilate_mask():
    input_mask = np.zeros((50, 50), np.float32)
    input_mask[:2, :] = 1
    expected_dilation = np.zeros((4, 50, 50), np.float32)
    for i in range(3):
        expected_dilation[i, :(5 - i), :] = 1
    expected_dilation[-1] = input_mask
    expected_dilation = expected_dilation[:, np.newaxis, :]

    assert np.array_equiv(expected_dilation, dilate_mask(input_mask, 3))


def test_make_bitmask():
    input_mask = np.zeros((5, 5), np.float32)
    input_mask[0, :] = 5
    input_mask[1, :] = 4

    expected_mask = np.zeros((5, 5), np.uint8)
    expected_mask[:1, :] = 1
    np.array_equiv(expected_mask, make_bitmask(input_mask, np.uint8))
