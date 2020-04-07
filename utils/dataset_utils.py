import numpy as np


BAYER_PATTERNS = ["RGGB", "BGGR", "GRBG", "GBRG"]
NORMALIZATION_MODE = ["crop", "pad"]


def bayer_unify(raw: np.ndarray, rgb: np.ndarray, input_pattern: str, target_pattern: str, mode: str) -> np.ndarray:
    """
    Convert a bayer raw image from one bayer pattern to another.

    Parameters
    ----------
    raw : np.ndarray in shape (H, W)
        Bayer raw image to be unified.
    input_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
        The bayer pattern of the input image.
    target_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
        The expected output pattern.
    mode: {"crop", "pad"}
        The way to handle submosaic shift. "crop" abandons the outmost pixels,
        and "pad" introduces extra pixels. Use "crop" in training and "pad" in
        testing.
    """
    if input_pattern not in BAYER_PATTERNS:
        raise ValueError('Unknown input bayer pattern!')
    if target_pattern not in BAYER_PATTERNS:
        raise ValueError('Unknown target bayer pattern!')
    if mode not in NORMALIZATION_MODE:
        raise ValueError('Unknown normalization mode!')
    if not isinstance(raw, np.ndarray) or len(raw.shape) != 2:
        raise ValueError('raw should be a 2-dimensional numpy.ndarray!')

    if not isinstance(rgb, np.ndarray) or len(rgb.shape) != 3:
        raise ValueError('rgb should be a 3-dimensional numpy.ndarray!')

    if input_pattern == target_pattern:
        h_offset, w_offset = 0, 0
    elif input_pattern[0] == target_pattern[2] and input_pattern[1] == target_pattern[3]:
        h_offset, w_offset = 1, 0
    elif input_pattern[0] == target_pattern[1] and input_pattern[2] == target_pattern[3]:
        h_offset, w_offset = 0, 1
    elif input_pattern[0] == target_pattern[3] and input_pattern[1] == target_pattern[2]:
        h_offset, w_offset = 1, 1
    else:  # This is not happening in ["RGGB", "BGGR", "GRBG", "GBRG"]
        raise RuntimeError('Unexpected pair of input and target bayer pattern!')

    if mode == "pad":
        out1 = np.pad(raw, [[h_offset, h_offset], [w_offset, w_offset]], 'reflect')
        out2_1 = np.pad(rgb[:,:,0], [[h_offset, h_offset], [w_offset, w_offset]], 'reflect')
        out2_2 = np.pad(rgb[:,:,1], [[h_offset, h_offset], [w_offset, w_offset]], 'reflect')
        out2_3 = np.pad(rgb[:,:,2], [[h_offset, h_offset], [w_offset, w_offset]], 'reflect')
        out2 = np.dstack((out2_1, out2_2, out2_3))
    elif mode == "crop":
        h, w = raw.shape
        out1 = raw[h_offset:h - h_offset, w_offset:w - w_offset]
        out2 = rgb[h_offset:h - h_offset, w_offset:w - w_offset, :]
    else:
        raise ValueError('Unknown normalization mode!')

    return out1, out2


def bayer_aug(raw: np.ndarray, rgb: np.ndarray, flip_h: bool, flip_w: bool, transpose: bool, input_pattern: str) -> np.ndarray:
    """
    Apply augmentation to a bayer raw image.

    Parameters
    ----------
    raw : np.ndarray in shape (H, W)
        Bayer raw image to be augmented. H and W must be even numbers.
    flip_h : bool
        If True, do vertical flip.
    flip_w : bool
        If True, do horizontal flip.
    transpose : bool
        If True, do transpose.
    input_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
        The bayer pattern of the input image.
    """

    if input_pattern not in BAYER_PATTERNS:
        raise ValueError('Unknown input bayer pattern!')
    if not isinstance(raw, np.ndarray) or len(raw.shape) != 2:
        raise ValueError('raw should be a 2-dimensional numpy.ndarray')
    if raw.shape[0] % 2 == 1 or raw.shape[1] % 2 == 1:
        raise ValueError('raw should have even number of height and width!')

    if not isinstance(rgb, np.ndarray) or len(rgb.shape) != 3:
        raise ValueError('rgb should be a 3-dimensional numpy.ndarray')
    if rgb.shape[0] % 2 == 1 or rgb.shape[1] % 2 == 1:
        raise ValueError('rgb should have even number of height and width!')

    aug_pattern, target_pattern = input_pattern, input_pattern

    out1 = raw
    out2 = rgb
    if flip_h:
        out1 = out1[::-1, :]
        out2 = out2[::-1, :, :]
        aug_pattern = aug_pattern[2] + aug_pattern[3] + aug_pattern[0] + aug_pattern[1]
    if flip_w:
        out1 = out1[:, ::-1]
        out2 = out2[:, ::-1, :]
        aug_pattern = aug_pattern[1] + aug_pattern[0] + aug_pattern[3] + aug_pattern[2]
    if transpose:
        out1 = out1.T
        out2_1 = out2[:,:,0].T
        out2_2 = out2[:,:,1].T
        out2_3 = out2[:,:,2].T
        out2 = np.dstack((out2_1, out2_2, out2_3))
        aug_pattern = aug_pattern[0] + aug_pattern[2] + aug_pattern[1] + aug_pattern[3]

    out1, out2 = bayer_unify(out1, out2, aug_pattern, target_pattern, "crop")
    return out1, out2
    
class Augment_Bayer:
    """
    Inputs:
        raw: shape (H,W,1)
        rgb: shape (H,W,3)
    Outputs:
        raw: shape (H,W,1)
        rgb: shape (H,W,3)
    """
    def __init__(self):
        pass
    def transform0(self, raw, rgb):
        return raw.copy(), rgb.copy()
    def transform1(self, raw, rgb):
        raw_flip_v, rgb_flip_v = bayer_aug(raw[...,0], rgb, flip_h=True, flip_w=False, transpose=False, input_pattern="RGGB")
        raw_flip_v = raw_flip_v[..., np.newaxis]
        return raw_flip_v.copy(), rgb_flip_v.copy()
    def transform2(self, raw, rgb):
        raw_flip_h, rgb_flip_h = bayer_aug(raw[...,0], rgb, flip_h=False, flip_w=True, transpose=False, input_pattern="RGGB")
        raw_flip_h = raw_flip_h[..., np.newaxis]
        return raw_flip_h.copy(), rgb_flip_h.copy()
    def transform3(self, raw, rgb):
        raw_flip_h, rgb_flip_h = bayer_aug(raw[...,0], rgb, flip_h=False, flip_w=False, transpose=True, input_pattern="RGGB")
        raw_flip_h = raw_flip_h[..., np.newaxis]
        return raw_flip_h.copy(), rgb_flip_h.copy()

