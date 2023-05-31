
import matplotlib.pyplot as plt
import torch as th
from torch.nn.functional import interpolate

from .celeste_level.level import LevelRenderer


def one_hot_to_image(
    onehot,
    norm=plt.Normalize(vmin=0, vmax=LevelRenderer.max_idx),
    cm=plt.cm.nipy_spectral,
):
    return cm(norm(onehot[0].argmax(0).cpu()))


def color_to_idx(img):
    return LevelRenderer.color_to_idx(img)


def idx_to_onehot(ordinal, device):
    return (
        th.nn.functional.one_hot(ordinal.to(th.int64), LevelRenderer.max_idx + 1)
        .transpose(0, 2)
        .transpose(1, 2)[None, :, :, :]
        .to(device)
    )


# def upscale_np(img, shape: List[int]):
#     newimg = img.copy()
#     kernel = np.ones((3, 3))*1.5
#     newimg[:, 2:] = cv2.dilate(newimg, kernel, iterations=1)
#     return interpolate(newimg, shape, mode="area")

# def upscale(img, shape: List[int]):
#     newimg = img.clone()
#     kernel = th.ones(3, 3, device=img.device)*1.5
#     newimg[:, 2:] = dilation(newimg[:, 2:], kernel)
#     return interpolate(newimg, shape, mode="nearest")


# def upscale(img, shape):
#     return celeste_downsampling(1, [[shape[0]/img.shape[-2],
#                                      shape[1]/img.shape[-1]]]
# ,idx_to_onehot(img[0].argmax(0),img.device))[0]


def upscale(img, shape):
    return interpolate(img, shape, mode="bilinear", align_corners=False)
