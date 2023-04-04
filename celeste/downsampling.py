import torch
from torch.nn.functional import interpolate
from torch.nn import Softmax
import torch as th
from .level import LevelRenderer

def celeste_downsampling(num_scales, scales, image, token_list = [1,
                                                                  0,
                                                                  2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        20], interp_args={'mode':'bilinear', 'align_corners':False}):
    """
    Special Downsampling Method designed for Super Mario Bros. Token based levels.

    num_scales : number of scales the image is scaled down to.
    scales : downsampling scales. Should be an array tuples (scale_x, scale_y) of length num_scales.
    image : Original level to be scaled down. Expects a torch tensor.
    token_list : list of tokens appearing in the image in order of channels from image.
    """

    scaled_list = []
    for sc in range(num_scales):
        scale_v = scales[sc][0]
        scale_h = scales[sc][1]

        # Initial downscaling of one-hot level tensor is normal bilinear scaling
        bil_scaled = interpolate(image.float(), (int(image.shape[-2] * scale_v), int(image.shape[-1] * scale_h)),**interp_args)

        # create masks for ordinal
        tmp = bil_scaled[0]
        zeros = th.zeros((tmp.shape[-2], tmp.shape[-1]), dtype=th.int64, device=image.device)

        # overwrite low prio values with high prio
        for i in token_list:
            zeros = th.where(tmp[i] > 0, (tmp[i] > 0)*i, zeros)

        # convert back to one hot
        before_softmax = th.nn.functional.one_hot(zeros, LevelRenderer.max_idx+1).transpose(0,2).transpose(1,2)[None, :, :, :]
        
        # apply softmax (more similar to generator output, same result on argmax)
        img_scaled = Softmax(dim=1)(30*before_softmax.float())

        scaled_list.append(img_scaled)

    scaled_list.reverse()
    return scaled_list
