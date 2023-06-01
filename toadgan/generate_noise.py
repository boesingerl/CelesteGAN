from typing import List

# Code based on https://github.com/tamarott/SinGAN
import torch


def generate_spatial_noise(size: List[int], device: torch.device):
    """Generates a noise tensor. Currently uses torch.randn."""
    # noise = generate_noise([size[0], *size[2:]], *args, **kwargs)
    # return noise.expand(size)

    return torch.randn(size, device=device)
