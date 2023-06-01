---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import torch as th
import os
import torch
import wandb
import PIL
import networkx as nx
import kornia
import cv2
import random

from tqdm.auto import tqdm
from glob import glob
from torch.nn.functional import interpolate
from torch.nn import Softmax

from itertools import chain
from toadgan.celeste.onnx_read import LevelGen
from typing import Tuple

from toadgan.celeste.downsampling import celeste_downsampling
from toadgan.celeste.image import color_to_idx, one_hot_to_image
from toadgan.models import init_models, reset_grads, restore_weights
from toadgan.models.generator import Level_GeneratorConcatSkip2CleanAdd
from toadgan.train_single_scale import train_single_scale
from toadgan.config import Config
from toadgan.celeste.celeste_level.level import LevelRenderer
from toadgan.generate_samples import generate_samples
from toadgan.celeste.image import idx_to_onehot, upscale
from toadgan.celeste.celeste_level.attributes import map_from_tensors
from toadgan.celeste.celeste_level.level import LevelEncoder

import matplotlib.patheffects as path_effects
from pathlib import Path
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
```

## Read levels, create config

```python
opt = Config().parse_args("--input_dir input --input_name celeste_2x.txt "
                           "--num_layer 4 --alpha 100 --niter 150 --nfc 64 --game celeste".split())

opt.scales = [0.75, 0.4]
opt.token_list = list(range(LevelRenderer.max_idx+1))
opt.nc_current = LevelRenderer.max_idx+1
opt.plot_scale = False
opt.log_partial = True
opt.out = 'data/output'

level_paths = sorted(glob('data/input_img/*.png'))
imgs =  [color_to_idx(np.array(PIL.Image.open(path))).astype('uint8') for path in level_paths]

multiple_reals = []

for ordinal in imgs:
    
    real = th.nn.functional.one_hot(th.tensor(ordinal, dtype=th.int64), LevelRenderer.max_idx+1).transpose(0,2).transpose(1,2)[None, :, :, :].to(opt.device)

    scales = [[x, x] for x in opt.scales]
    opt.num_scales = len(scales)

    scaled_list = celeste_downsampling(opt.num_scales, scales, real, interp_args=dict(mode='area'))
    tmp_reals = [*scaled_list, real.float()]
    
    multiple_reals.append(tmp_reals)
```

## Train one SinGan instance per level

```python
def get_tags(opt):
    """ Get Tags for logging from input name. Helpful for wandb. """
    return [opt.input_name.split(".")[0]]

for i, reals in enumerate(tqdm(multiple_reals)):
    
    run = wandb.init(project="celeste_redo", tags=get_tags(opt),
                 config=opt, dir=opt.out)

    
    generators = []
    noise_maps = []
    noise_amplitudes = []
    input_from_prev_scale = torch.zeros_like(reals[0])

    stop_scale = len(reals)
    opt.stop_scale = stop_scale

    wandb.log({f"real{i}": wandb.Image(one_hot_to_image(reals[-1]))}, commit=False)
    
    lvlname = f'data/may30/lvl_{i:03d}'
    # Training Loop
    for current_scale in range(0, stop_scale):
        
        opt.outf = "%s/%d" % (lvlname, current_scale)
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass
        
        try:
            os.makedirs("%s/state_dicts" % (lvlname), exist_ok=True)
        except OSError:
            pass
        try:
            os.makedirs(lvlname)
        except OSError:
            pass
    

        # If we are seeding, we need to adjust the number of channels
        if current_scale < (opt.token_insert + 1):  # (stop_scale - 1):
            opt.nc_current = len(token_group)

        # Initialize models
        D, G = init_models(opt)
        # If we are seeding, the weights after the seed need to be adjusted
        if current_scale == (opt.token_insert + 1):  # (stop_scale - 1):
            D, G = restore_weights(D, G, current_scale, opt)

        # Actually train the current scale
        z_opt, input_from_prev_scale, G = train_single_scale(D,  G, reals, generators, noise_maps,
                                                             input_from_prev_scale, noise_amplitudes, opt)
        
        # Reset grads and save current scale
        G = reset_grads(G, False)
        G.eval()
        D = reset_grads(D, False)
        D.eval()

        generators.append(G)
        noise_maps.append(z_opt)
        noise_amplitudes.append(opt.noise_amp)

    torch.save(noise_maps, "%s/noise_maps.pth" % (lvlname))
    torch.save(generators, "%s/generators.pth" % (lvlname))
    torch.save(reals, "%s/reals.pth" % (lvlname))
    torch.save(noise_amplitudes, "%s/noise_amplitudes.pth" % (lvlname))
    torch.save(input_from_prev_scale, "%s/input_from_prev_scale.pth" % (lvlname))
    
    torch.save(opt.num_layer, "%s/num_layer.pth" % (lvlname))
    torch.save(opt.token_list, "%s/token_list.pth" % (lvlname))
    
    
    fulldict = {
        "noise_maps": noise_maps,
        "generators":generators,
        "reals":reals,
        "noise_amplitudes":noise_amplitudes,
        "lvlname":lvlname
    }
    
    torch.save(fulldict, "%s/fulldict.pth" % (lvlname))
    
    wandb.save("%s/*.pth" % lvlname)

    torch.save(G.state_dict(), "%s/state_dicts/G_%d.pth" % (lvlname, current_scale))
    wandb.save("%s/state_dicts/*.pth" % lvlname)

```

## Generate samples

```python
def generate_level(path, maskpath=None, masktype='img', seed=None, device='cuda', replace_noise=False):

    if seed is not None:
        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if device == 'cpu':
            torch.use_deterministic_algorithms(True)
        
        
    opt = Config().parse_args("--input_dir input --input_name celeste_single_image.txt "
                           "--num_layer 4 --alpha 100 --niter 500 --nfc 64 --game celeste".split())
    
    opt.device = device
    opt.scales = [0.7, 0.4]
    opt.token_list = list(range(LevelRenderer.max_idx+1))
    opt.nc_current = LevelRenderer.max_idx+1
    opt.log_all = True
    opt.log_progress = False

    opt.log_interp = []
    opt.log_noise = []
    opt.log_out = []
    
                
    
    dic = torch.load(path)

    noise_maps = dic['noise_maps']
    generators = dic['generators']
    reals = dic['reals']
    noise_amplitudes = dic['noise_amplitudes']
    
     
    use_reals = reals
    use_maps = noise_maps
    
    if maskpath is not None:
        if masktype == "img":
            level = (plt.imread(maskpath)*255).astype(int)
    
            if level.shape[-1] == 3:
                level = np.dstack((level, np.ones((level.shape[0], level.shape[1]))*255))
    
            ordinal = color_to_idx(level)
            mask = th.nn.functional.one_hot(
                th.tensor(ordinal, dtype=th.int64), LevelRenderer.max_idx+1).transpose(0,2).transpose(1,2)[None, :, :, :].to(opt.device).float()
            mask[:, 0, :, :] = 0
        elif masktype == "array":
            with gzip.open(maskpath, "rb") as handle:
                mask = np.array(json.loads(handle.read().decode()), dtype="float32")
                mask[0, 0] = 0
                mask[0, 20] = mask[0, 20] * 2
                mask[0, 32] = mask[0, 32] * 2
                mask = th.tensor(mask, device=opt.device)
        else:
            raise Exception('Type has to be img or array')
    else:
        mask = None

   

    all_samples = generate_samples(generators, 
                     use_maps,
                     use_reals,
                     noise_amplitudes,
                     opt,
                     in_s=mask,
                     save_dir="arbitrary_random_samples",
                     num_samples=1,
                     replace_noise=replace_noise
                    )
    
    samp = all_samples[0]
    
    if maskpath is not None:
        mask_scaled = celeste_downsampling(1, [[samp.shape[-2]/mask.shape[-2], samp.shape[-1]/mask.shape[-1]]], mask)[0]

        samp[0,20] = mask_scaled[0, 20]*20
        samp[0,32] = mask_scaled[0, 32]*20

        yfin, xfin = [torch.median(x).item() for x in torch.where(samp[0].argmax(0) == 32)]
        ysta, xsta = [torch.median(x).item() for x in torch.where(samp[0].argmax(0) == 20)]
    
        return opt,force_path_th(all_samples[0])
    
    return opt, all_samples[0]

def plot_generation(ident, maskpath='celeste/manual_level/masksmall.png', img_last=False, masktype='img', seed=None, device='cuda', replace_noise=False):

    path = all_level_paths[ident]
    optmp, lvl = generate_level(path, maskpath=maskpath, masktype=masktype, seed=seed, device=device, replace_noise=replace_noise)
    
    len_ = len(optmp.log_noise)
    fig, axs = plt.subplots(ncols=3, nrows=len_, figsize=(9, len_*2))
    
    for i, (axn, axo, axi) in enumerate(axs):
        img = optmp.log_noise[i]
        axn.imshow(img)
        axn.set(title=f'Noise {i} {img.shape[-3:-1]}')
        axn.axis('off')
        
        img = optmp.log_out[i]
        axo.imshow(img)
        txt = axo.set_title(f'Output {i} {img.shape[-3:-1]}' if i < len_-1 else f'Final Output {img.shape[-3:-1]}')
        txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                       path_effects.Normal()])
        axo.axis('off')
        
        
        if len(optmp.log_interp) > i:
            img = optmp.log_interp[i]
            axi.imshow(img)
            axi.set(title=f'Interpolated {i} {img.shape[-3:-1]}')
            
        else:
            if img_last:
                # img = plt.imread(level_paths[ident])
                # axi.imshow(img)   
                axi.set(title=f'Original level {img.shape[-3:-1]}')
        axi.axis('off')
                
    for i, (ax1, ax2) in enumerate(zip(axs.flatten(), axs.flatten()[1:])):
    
        if i < len(axs.flatten())-2:
            con = ConnectionPatch(
                # in axes coordinates
                xyA=(1, 0.5) if i%3 != 2 else (0,0), coordsA=ax1.transAxes,
                # x in axes coordinates, y in data coordinates
                xyB=(0.0, 0.5) if i%3 != 2 else (1,1), coordsB=ax2.transAxes,
                arrowstyle="-|>")

            ax1.add_artist(con)
    
    plt.axis('off')
    plt.subplots_adjust(wspace=0.2, hspace=0.4) 
    
    return fig, lvl
                
def graph_from_img(bw):
    
    G = nx.Graph()

    for y, row in enumerate(bw):

        for x, val in enumerate(row):

            G.add_node((x,y))

            try:
                if bw[y, x-1].item() == val.item() and val.item() == 0:
                
                    if not neighbours_wall(bw, (x-1,y)) and not neighbours_wall(bw, (x,y)):
                        G.add_edge((x,y), (x-1, y), weight=1)
                    else:
                        G.add_edge((x,y), (x-1, y), weight=50)
                        
                else:
                    G.add_edge((x,y), (x-1, y), weight=100)
            except:
                pass

            try:
                if bw[y-1, x].item() == val.item() and val.item() == 0:
                    if not neighbours_wall(bw, (x, y-1)) and not neighbours_wall(bw, (x,y)):
                        G.add_edge((x,y), (x, y-1), weight=1)
                    else:
                        G.add_edge((x,y), (x, y-1), weight=50)
                        
                else:
                    G.add_edge((x,y), (x-1, y), weight=100)
            except:
                pass
            
    return G

def neighbours_wall(img, point):
    
    x,y = point
    
    try:
        if img[y-1, x] == 1:
            return True
    except:
        pass
    
    try:
        if img[y+1, x] == 1:
            return True
    except:
        pass
    
    try:
        if img[y, x-1] == 1:
            return True
    except:
        pass
    
    try:
        if img[y, x+1] == 1:
            return True
    except:
        pass
    
    return False

def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5



def force_path_th(sample):

    sample = sample.clone()
    
    bw = sample[0].argmax(0).cpu()
    bw[bw > 1] = 0

    G = graph_from_img(bw)

    yfin, xfin = [torch.median(x).item() for x in torch.where(sample[0].argmax(0) == 32)]
    ysta, xsta = [torch.median(x).item() for x in torch.where(sample[0].argmax(0) == 20)]

    path = nx.astar_path(G, (xfin, yfin), (xsta, ysta), heuristic=dist, weight="weight")

    
    def set_path(img, point):
        x,y = point
        
        
        def tryset(img, x, y):
            try:
                img[0, 1, y, x] = 0
            except:
                pass
            
        tryset(img, x, y)
        tryset(img, x-1, y)
        tryset(img, x+1, y)
        tryset(img, x, y-1)
        tryset(img, x, y+1)
        
    for p in path:
        set_path(sample, p)
        
    return sample
```

## Create onnx generators

```python
for i, path in enumerate(sorted(chain(glob('data/may24/lvl_*/fulldict.pth'),glob('data/may30/lvl_*/fulldict.pth')))):
    
    dic = torch.load(path, map_location='cpu') 
    
    gentest = LevelGen(dic['generators'], dic['noise_maps'], dic['reals'], dic['noise_amplitudes'])    
    
    gentest.to_json(f'data/generators2/{i:03d}.json.gz')  
```

## Plot the various stages

```python
all_level_paths = sorted(glob('data/may24/lvl_*/fulldict.pth'))

fig, lvl = plot_generation(5, img_last=True, maskpath=None)

map_ = map_from_tensors([lvl.cpu().numpy()])
LevelEncoder.write_level('data/demo_level.bin', map_)
fig.savefig('data/levelout.png', transparent=True)
```

## Save upscale onnx (since there's no exact same method in numpy)

```python
def interp_shape(tens, shape: Tuple[int, int]):
    return interpolate(tens, shape, mode="bilinear", align_corners=False)

script_upscale = torch.jit.script(interp_shape)
torch.onnx.export(script_upscale,
                  (th.tensor(masknp), (8,10)),
                  'upscale.onnx',
                  input_names=['image', 'shape0', 'shape1'],
                  output_names=['out'],
                  opset_version=18,
                  dynamic_axes={"image": {0: "batch", 2: "height", 3:"width"}})
```
