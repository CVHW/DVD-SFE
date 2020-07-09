import random
import numpy as np
import torch


def get_patch(*args, patch_size=256):
    """
    Get patch from an image blur sharp gt
    may be the shape of sharp is not euqal to the shape of blur/gt
    """
    ih, iw, c = args[0].shape

    ix = random.randrange(0, iw - patch_size + 1)
    iy = random.randrange(0, ih - patch_size + 1)

    ret = []

    for arg in args:
        if arg.shape == (ih, iw, c):
            ret.append(arg[iy:iy + patch_size, ix:ix + patch_size, :])
        else:
            ih_sharp, iw_sharp, _ = arg.shape
            ix_sharp = random.randrange(0, iw_sharp - patch_size + 1)
            iy_sharp = random.randrange(0, ih_sharp - patch_size + 1)
            ret.append(arg[iy_sharp:iy_sharp + patch_size, ix_sharp:ix_sharp + patch_size, :])
    return ret


def np2Tensor(*args, rgb_range=255):
    def _np2tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2tensor(a) for a in args]


def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = np.rot90(img)

        return img

    return [_augment(a) for a in args]
