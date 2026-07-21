import numbers
import PIL.Image
import cv2
import numpy as np
import PIL
import torch


def _is_tensor_clip(clip):
    return torch.is_tensor(clip) and clip.ndimension() == 4


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def resize_clip(clip, size, interpolation='bilinear'):
    # NOTE: consider cv2.INTER_AREA (or PIL) for better downscaling quality.
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[0], size[1]
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        elif interpolation == 'area':
            np_inter = cv2.INTER_AREA
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = [
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.BILINEAR
        else:
            pil_inter = PIL.Image.NEAREST
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return scaled


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow


def normalize(clip, mean, std, inplace=False):
    """
    clip: C N H W
    """
    if not _is_tensor_clip(clip):
        raise TypeError('tensor is not a torch clip.')

    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

    return clip

def bnormalize(clip, mean, std, inplace=False):
    """
    clip: B C N H W
    """
    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.sub_(mean[None, :, None, None, None]).div_(std[None, :, None, None, None])

    return clip


def denormalize(clip, mean, std, inplace=False):
    """
    clip: B C N H W
    """

    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = [-m/s for m, s in zip(mean, std)]
    std = [1/s for s in std]
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)

    clip.sub_(mean[None, :, None, None, None]).div_(std[None, :, None, None, None])

    return clip

def denormalize_hands(clip, mean, std, inplace=False):
    """
    clip: B N J 3
    """

    if not inplace:
        clip = clip.clone()
    dtype = clip.dtype
    mean = [-m/s for m, s in zip(mean, std)]
    std = [1/s for s in std]
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)

    clip.sub_(mean[None, None, None, :]).div_(std[None, None, None, :])

    return clip


def denormalize_keypoints(clip, mean, std, inplace=False):
    """
    clip: B T J 3
    mean/std: J 3 or 3
    """
    if not inplace:
        clip = clip.clone()

    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)

    if mean.ndim == 1:
        clip.mul_(std[None, None, None, :]).add_(mean[None, None, None, :])
    else:
        clip.mul_(std[None, None, :, :]).add_(mean[None, None, :, :])

    return clip

def denormalize_cam(clip, mean, std, inplace=False):
    """
    clip: B N 9
    """
    if not inplace:
        clip = clip.clone()
    dtype = clip.dtype
    mean = [-m/s for m, s in zip(mean, std)]
    std = [1/s for s in std]
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)

    clip.sub_(mean[None, None, :]).div_(std[None, None, :])

    return clip
