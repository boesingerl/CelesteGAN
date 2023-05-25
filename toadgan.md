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

from tqdm.auto import tqdm
from glob import glob
from torch.nn.functional import interpolate
from torch.nn import Softmax

from toadgan.main import get_tags
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
                           "--num_layer 4 --alpha 100 --niter 1500 --nfc 64 --game celeste".split())

opt.scales = [0.75, 0.4]
opt.token_list = list(range(LevelRenderer.max_idx+1))
opt.nc_current = LevelRenderer.max_idx+1
opt.plot_scale = False
opt.log_partial = True
opt.out = 'data/output'

level_paths = sorted(glob('data/in_celeste/allnopad/*.png'))
imgs =  [color_to_idx(np.array(PIL.Image.open(path))).astype('uint8') for path in level_paths]
#imgs = [upscale(idx_to_onehot(torch.tensor(img), opt.device).float(), [2*x for x in img.shape])[0].argmax(0) for img in imgs]

multiple_reals = []

for ordinal in imgs:
    
    real = th.nn.functional.one_hot(th.tensor(ordinal, dtype=th.int64), LevelRenderer.max_idx+1).transpose(0,2).transpose(1,2)[None, :, :, :].to(opt.device)

    scales = [[x, x] for x in opt.scales]
    opt.num_scales = len(scales)

    scaled_list = celeste_downsampling(opt.num_scales, scales, real, interp_args=dict(mode='area'))
    tmp_reals = [*scaled_list, real.float()]
    
    multiple_reals.append(tmp_reals)
```

```python
# plt.imshow(one_hot_to_image(multiple_reals[0][-1]))
```

## Train one SinGan instance per level

```python
for i, reals in enumerate(tqdm(multiple_reals[:5])):
    
    run = wandb.init(project="celeste_redo", tags=get_tags(opt),
                 config=opt, dir=opt.out)

    
    generators = []
    noise_maps = []
    noise_amplitudes = []
    input_from_prev_scale = torch.zeros_like(reals[0])

    stop_scale = len(reals)
    opt.stop_scale = stop_scale

    wandb.log({f"real{i}": wandb.Image(one_hot_to_image(reals[-1]))}, commit=False)
    
    lvlname = f'data/may24/lvl_{i:03d}'
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

```

```python
def generate_level(path, maskpath=None):
    opt = Config().parse_args("--input_dir input --input_name celeste_single_image.txt "
                           "--num_layer 4 --alpha 100 --niter 500 --nfc 64 --game celeste".split())

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
        level = (plt.imread(maskpath)*255).astype(int)

        if level.shape[-1] == 3:
            level = np.dstack((level, np.ones((level.shape[0], level.shape[1]))*255))

        ordinal = color_to_idx(level)
        mask = th.nn.functional.one_hot(
            th.tensor(ordinal, dtype=th.int64), LevelRenderer.max_idx+1).transpose(0,2).transpose(1,2)[None, :, :, :].to(opt.device).float()
        mask[:, 0, :, :] = 0
    else:
        mask = None

   

    all_samples = generate_samples(generators, 
                     use_maps,
                     use_reals,
                     noise_amplitudes,
                     opt,
                     in_s=mask,
                     save_dir="arbitrary_random_samples",
                     num_samples=1)
    
    samp = all_samples[0]
    
    if maskpath is not None:
        mask_scaled = celeste_downsampling(1, [[samp.shape[-2]/mask.shape[-2], samp.shape[-1]/mask.shape[-1]]], mask)[0]

        samp[0,20] = mask_scaled[0, 20]*20
        samp[0,32] = mask_scaled[0, 32]*20

        yfin, xfin = [torch.median(x).item() for x in torch.where(samp[0].argmax(0) == 32)]
        ysta, xsta = [torch.median(x).item() for x in torch.where(samp[0].argmax(0) == 20)]
    
        return opt,force_path(all_samples[0])
    
    return opt, all_samples[0]

def plot_generation(ident, maskpath='celeste/manual_level/masksmall.png', img_last=False):

    path = all_level_paths[ident]
    optmp, lvl = generate_level(path, maskpath=maskpath)
    
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
                img = plt.imread(level_paths[ident])
                axi.imshow(img)   
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



def force_path(sample):

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

```python

all_level_paths = sorted(glob('data/may24/lvl_*/fulldict.pth'))

fig, lvl = plot_generation(1, img_last=True, maskpath=None)

map_ = map_from_tensors([lvl.cpu().numpy()])
LevelEncoder.write_level('data/demo_level.bin', map_)
fig.savefig('data/levelout.png', transparent=True)
```

```python


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

# Draw a simple arrow between two points in axes coordinates
# within a single axes.
xyA = (0.2, 0.2)
xyB = (0.8, 0.8)
coordsA = "data"
coordsB = "data"
con = ConnectionPatch(xyA, xyB, coordsA, coordsB,
                      arrowstyle="-|>", shrinkA=5, shrinkB=5,
                      mutation_scale=20, fc="w")
ax1.plot([xyA[0], xyB[0]], [xyA[1], xyB[1]], "o")
ax1.add_artist(con)

# Draw an arrow between the same point in data coordinates,
# but in different axes.
xy = (0.3, 0.2)
con = ConnectionPatch(
    xyA=xy, coordsA=ax2.transData,
    xyB=xy, coordsB=ax1.transData,
    arrowstyle="->", shrinkB=5)
fig.add_artist(con)

# Draw a line between the different points, defined in different coordinate
# systems.
con = ConnectionPatch(
    # in axes coordinates
    xyA=(0.6, 1.0), coordsA=ax2.transAxes,
    # x in axes coordinates, y in data coordinates
    xyB=(0.0, 0.2), coordsB=ax2.transAxes,
    arrowstyle="-")
ax2.add_artist(con)
```

```python
fig, axs = plt.subplots(ncols=3, nrows=3)

for i, (ax1, ax2) in enumerate(zip(axs.flatten(), axs.flatten()[1:])):
    
    con = ConnectionPatch(
        # in axes coordinates
        xyA=(1, 0.5) if i%3 != 2 else (0,0), coordsA=ax1.transAxes,
        # x in axes coordinates, y in data coordinates
        xyB=(0.0, 0.5) if i%3 != 2 else (1,1), coordsB=ax2.transAxes,
        arrowstyle="-|>")
    
    ax1.add_artist(con)
```

```python

```

## To onnx

```python
all_level_paths = np.random.choice(glob('apr1/lvl_*/fulldict.pth'), size=1)

path = all_level_paths[0]

maskpath='celeste/manual_level/masksmall.png'
```

```python
opt = Config().parse_args("--input_dir input --input_name celeste_single_image.txt "
                       "--num_layer 4 --alpha 100 --niter 500 --nfc 64 --game celeste".split())

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
    level = (plt.imread(maskpath)*255).astype(int)

    if level.shape[-1] == 3:
        level = np.dstack((level, np.ones((level.shape[0], level.shape[1]))*255))

    ordinal = color_to_idx(level)
    mask = th.nn.functional.one_hot(
        th.tensor(ordinal, dtype=th.int64), LevelRenderer.max_idx+1).transpose(0,2).transpose(1,2)[None, :, :, :].to(opt.device).float()
    mask[:, 0, :, :] = 0
else:
    mask = None



all_samples = generate_samples(generators, 
                 use_maps,
                 use_reals,
                 noise_amplitudes,
                 opt,
                 in_s=mask,
                 save_dir="arbitrary_random_samples",
                 num_samples=1)
```

```python
dic_np = {'noise_maps': [x.cpu().numpy() for x in dic['noise_maps']],
          'reals': [x.cpu().numpy() for x in dic['reals']],
          'noise_amplitudes': [np.asarray(x if isinstance(x, int) else x.cpu()) for x in dic['noise_amplitudes']],
          'lvlname':dic['lvlname']
         }
         
```

```python
with open('dic_np.pkl', 'wb') as handle:
    pickle.dump(dic_np, handle)
```

```python
dic['noise_amplitudes']
```

```python
from typing import Tuple

def interp_shape(tens, shape: Tuple[int, int]):
    return interpolate(tens, shape,mode="nearest")

script_upscale = torch.jit.script(interp_shape)
torch.onnx.export(script_upscale,
                  (mask, (8,10)),
                  'upscale.onnx',
                  input_names=['image', 'shape0', 'shape1'],
                  output_names=['out'],
                  opset_version=18,
                  dynamic_axes={"image": {0: "batch", 2: "height", 3:"width"}})
```

```python
from generate_noise import generate_spatial_noise
from typing import List, Any, Tuple

from tap import Tap
from typing import List, Literal, Optional, Union

import onnxruntime
import onnxruntime as ort
import numpy as np
import io

import base64
import sys
sys.getdefaultencoding()

from contextlib import redirect_stdout
import json
```

```python
class Upscaler:
    
    def __init__(self, path='upscale.onnx'):
        self.sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        
        self.kernel = np.ones((3, 3))*1.5
        self.dilate = lambda img,a: cv2.dilate(img, kernel, iterations=1)
        applied = np.apply_over_axes(dilate, masknp[0], 0)
        
    def __call__(self, image, shape):
        
        newimg = image.copy()
        kernel = np.ones((3, 3))*1.5
        newimg[:, 2:] = np.apply_over_axes(self.dilate, newimg[0, 2:], 0)[None,:,:,:]

        return self.sess.run(None, {'image': newimg, 'shape0':np.array(shape[0]), 'shape1':np.array(shape[1])})[0]
```

```python
masknp = mask.cpu().numpy()
scaler = Upscaler()
```

```python
scaled = scaler(masknp, (masknp.shape[-2]*2, masknp.shape[-1]*2))

masknp.shape, scaled.shape
```

```python
import gzip

class Generators:
    
    def __init__(self, generators, noise_maps):
        self.bytes = []
        for generator, noise_map in zip(generators, noise_maps):
            bytio = io.BytesIO()
            
            with redirect_stdout(None):
                torch.onnx.export(generator,
                      (noise_map,
                       noise_map,
                       1),
                      bytio,
                      input_names=['x','y','temperature'],
                      output_names=['output'])
            
            self.bytes.append(bytio)
            
    def to_json(self,path):
        with gzip.open(path, 'wb') as handle:
            handle.write(json.dumps({'bytes': [base64.b64encode(v.getvalue()).decode() for v in self.bytes]}).encode('utf-8'))
            
    def read_json(path):
        with gzip.open('gentest.json.gz', 'rb') as handle:
            read_gen = [ort.InferenceSession(base64.b64decode(allbytes), providers=['CPUExecutionProvider']) for allbytes in json.loads(handle.read())['bytes']]
            
        return read_gen
            
gens = Generators(generators, noise_maps)
gens.to_json('gentest.json.gz')
```

```python
sessions = Generators.read_json('gentest.json.gz')
```

```python
def format_and_use_generator(curr_img, G_z, count, Z_opt, pad_noise, pad_image, noise_amp, G, opt):
    """ Correctly formats input for generator and runs it through. """
    
    if curr_img.shape != G_z.shape:
        G_z = scaler(G_z, curr_img.shape[-2:])
        
    curr_img = pad_noise(curr_img)  # Curr image is z in this case
    z_add = curr_img

    G_z = pad_image(G_z)
    
    z_in = noise_amp * z_add + G_z

    G_z = G.run(None, {'x':z_in, 'y':G_z, 'temperature':np.array(1)})[0]
    return G_z


def draw_concat(generators, noise_maps, reals, noise_amplitudes, in_s, pad_noise, pad_image, opt):
    """ Draw and concatenate output of the previous scale and a new noise map. """

    G_z = in_s
    
    noise_padding = 1 * opt.num_layer
    pad_noise = pad_noise(noise_padding)
    pad_image = pad_image(noise_padding)
    
    for count, (G, Z_opt, real_curr, real_next, noise_amp) in enumerate(
            zip(generators, noise_maps, reals, reals[1:], noise_amplitudes)):

        if count < opt.stop_scale:  # - 1):
            z = np.random.normal(size=[1,
                                       real_curr.shape[1],
                                       Z_opt.shape[2] - 2 * noise_padding,
                                       Z_opt.shape[3] - 2 * noise_padding]).astype('float32')

            
        G_z = format_and_use_generator(z, G_z, count, Z_opt,
                                       pad_noise, pad_image, noise_amp, G, opt)
                
    return G_z

```

```python
nmaps = [x.cpu() for x in noise_maps]
nreals = [x.cpu() for x in reals]
noise_test = np.zeros_like(noise_maps[0].cpu().numpy())
namps = [x if isinstance(x, int) else x.item() for x in noise_amplitudes]
```

```python
def padding_func(padsize):
    def pad_(img):
        return np.pad(img, ((0,), (0,), (padsize,), (padsize,)))
    return pad_
```

```python
test = draw_concat(sessions, nmaps, nreals, namps, noise_test,padding_func ,padding_func , opt)
plt.imshow(one_hot_to_image(torch.tensor(test)))
```

```python
%%time
ort_sess = ort.InferenceSession('gen0.onnx', providers=['CPUExecutionProvider'])

```

```python
%%timeit

outputs = ort_sess.run(None, {'x':x_np, 'y':y_np, 'temperature': np.array(1)})[0]
```

```python
%%timeit

ort_sess = ort.InferenceSession('gen0.onnx', providers=['CUDAExecutionProvider'])
outputs = ort_sess.run(None, {'x':x_np, 'y':y_np, 'temperature': np.array(1)})[0]
```

```python
%%timeit

generators[0].to('cpu')(x.to('cpu'),y.to('cpu'),temperature=1)
```

```python
x,y = noise_maps[0], noise_maps[0]
x_np, y_np = noise_maps[0].cpu().numpy(), noise_maps[0].cpu().numpy()
```

```python
import onnx

onnx.checker.check_model('gen0.onnx')
```

```python
for k in c.__dict__.keys():
    print(f"self.{k} = d['{k}']")
```

```python
for k in c.__dict__.keys():
    print(f"'{k}': self.{k},")
```

```python
def identity(x):
    return x

from generate_samples import Padder

class InferenceGen(torch.nn.Module):
    def __init__(self,
                 gen0: Level_GeneratorConcatSkip2CleanAdd,
                 gen1: Level_GeneratorConcatSkip2CleanAdd,
                 gen2: Level_GeneratorConcatSkip2CleanAdd,
                 noise_maps: Tuple[torch.tensor,torch.tensor,torch.tensor],
                 reals: Tuple[torch.tensor,torch.tensor,torch.tensor],
                 noise_amplitudes: Tuple[torch.tensor,torch.tensor,torch.tensor],
                 opt: ConfigTorch,
                 scale_v:float=1.0,
                 scale_h:float=1.0,
                 current_scale:int=0,
                 gen_start_scale:int=0,
                 num_samples:int=50,
                 render_images:bool=True,
                 save_tensors:bool=False,
                 save_dir="random_samples"):
        super().__init__()

        trace_gen = lambda gen: torch.jit.trace(gen,
                                   example_inputs=(torch.randn((1,33,64,64),device=opt.device),
                                                   torch.randn((1,33,64,64),device=opt.device)))
        self.gen0 = trace_gen(gen0)
        self.gen1 = torch.jit.trace(gen1,
                                   example_inputs=(torch.randn((1,33,64,64),device=opt.device),
                                                   torch.randn((1,33,64,64),device=opt.device)))
        self.gen2 = torch.jit.trace(gen2,
                                   example_inputs=(torch.randn((1,33,64,64),device=opt.device),
                                                   torch.randn((1,33,64,64),device=opt.device)))
        self.gens : Tuple["Level_GeneratorConcatSkip2CleanAdd",
                          "Level_GeneratorConcatSkip2CleanAdd",
                          "Level_GeneratorConcatSkip2CleanAdd"
                         ]= (self.gen0, self.gen1, self.gen2)
        self.noise_maps = noise_maps
        self.reals = reals
        self.noise_amplitudes = noise_amplitudes
        self.opt = opt
        self.scale_h = scale_h
        self.scale_v = scale_v
        self.current_scale = current_scale
        self.gen_start_scale = gen_start_scale
        self.num_samples = num_samples
        self.render_images = render_images
        self.save_tensors = save_tensors
        self.save_dir = save_dir

    def forward(self, in_s):
         # Holds images generated in current scale
        images_cur = []

        tqdm_wrap = identity

                           
        noise_maps = self.noise_maps
        reals = self.reals
        noise_amplitudes = self.noise_amplitudes
        opt = self.opt
        scale_h = self.scale_h
        scale_v = self.scale_v
        current_scale = self.current_scale
        gen_start_scale = self.gen_start_scale
        num_samples = self.num_samples
        render_images = self.render_images
        save_tensors = self.save_tensors
        save_dir = self.save_tensors

        final_tensors = []
        # Main sampling loop
        for sc, (Z_opt, noise_amp) in enumerate(zip(noise_maps, noise_amplitudes)):

            n_pad = int(1*opt.num_layer)

            m = Padder(int(n_pad))

            # Calculate shapes to generate
            if 0 < gen_start_scale <= current_scale:  # Special case! Can have a wildly different shape through in_s
                scale_v = in_s.shape[-2] / \
                    (noise_maps[gen_start_scale-1].shape[-2] - n_pad * 2)
                scale_h = in_s.shape[-1] / \
                    (noise_maps[gen_start_scale-1].shape[-1] - n_pad * 2)
                nzx = (Z_opt.shape[-2] - n_pad * 2) * scale_v
                nzy = (Z_opt.shape[-1] - n_pad * 2) * scale_h
            else:
                nzx = (Z_opt.shape[-2] - n_pad * 2) * scale_v
                nzy = (Z_opt.shape[-1] - n_pad * 2) * scale_h

            # Save list of images of previous scale and clear current images
            images_prev = images_cur
            images_cur = []

            channels = len(opt.token_list)


            # If in_s is none or filled with zeros reshape to correct size with channels
            if in_s is None:
                in_s = torch.zeros(reals[0].shape[0], channels,
                                   *reals[0].shape[2:]).to(opt.device)
            elif in_s.sum() == 0:
                in_s = torch.zeros(1, channels, in_s.shape[-2], in_s.shape[-1]).to(opt.device)

            # Generate num_samples samples in current scale
            for n in tqdm_wrap(torch.arange(0, num_samples, 1)):

                # Get noise image
                z_curr = generate_spatial_noise(
                    [1, channels, int(round(nzx)), int(round(nzy))], device=opt.device)
                z_curr = m(z_curr)

                # Set up previous image I_prev
                if (not images_prev) or current_scale == 0:  # if there is no "previous" image
                    I_prev = in_s
                else:

                    I_prev = images_prev[n]


                if opt.game != 'celeste':
                    I_prev = interpolate(I_prev, [int(round(nzx)), int(
                        round(nzy))], mode='bilinear', align_corners=False)
                else:

                    if sc == 0:
                        I_prev = upscale(I_prev, [int(round(nzx)), int(round(nzy))])

                    else:
                        I_prev = upscale(I_prev, [int(round(nzx)), int(round(nzy))])

                        # if opt.log_all:
                        #     opt.log_interp.append(one_hot_to_image(I_prev))

                I_prev = m(I_prev)

                # We take the optimized noise map Z_opt as an input if we start generating on later scales
                if current_scale < gen_start_scale:
                    z_curr = Z_opt


                token_list = opt.token_list

                z_in = noise_amp * z_curr + I_prev
                
                if sc == 0:
                    I_curr = self.gen0(z_in.detach(), I_prev)
                elif sc == 1:
                    I_curr = self.gen1(z_in.detach(), I_prev)
                else:
                    I_curr = self.gen2(z_in.detach(), I_prev)
                    
                    
                

                # Append current image
                images_cur.append(I_curr)
                    
                if current_scale == len(reals) - 1:

                    if opt.game == 'celeste':
                        final_tensors.append(I_curr.detach())
                    

            # Go to next scale
            current_scale += 1

        return final_tensors[0]

inference_gen = torch.jit.script(InferenceGen(*generators, tuple(noise_maps), tuple(reals), tuple(noise_amplitudes), opt=c))
```

```python
%%timeit

plt.imshow(one_hot_to_image(inference_gen(mask)))
```

```python
torch.onnx.export(inference_gen, mask, "test.onnx")
```

```python
nois
```

```python
%%timeit

plt.imshow(one_hot_to_image(generate_samples(generators, noise_maps, reals, noise_amplitudes, opt=c)[0]))
```

```python
f = MyScriptModule(*generators, tuple(noise_maps), tuple(reals), tuple(noise_amplitudes), opt=c)
```

```python
plt.imshow(one_hot_to_image(f(mask)))
```

```python

```

```python

```

```python
type(c)
```

```python
noise_maps
```

```python
torch.jit.script(tuple(generators))
```

```python
import typing
```

```python
test = (MyScriptModule(tuple(generators), noise_maps, reals, noise_amplitudes, opt))
```

```python
type(test.gen0)
```

```python

```

```python
from typing import get_type_hints

get_type_hints(test.generators)
```

```python
test = (MyScriptModule(tuple(generators), noise_maps, reals, noise_amplitudes, opt))
```

```python
generators
```

```python
from generate_samples import generate_samples

from models import Level_GeneratorConcatSkip2CleanAdd
```

```python
def to_script(mask):
    return generate_samples(generators, 
             maps,
             reals,
             amps,
             opt,
             in_s=mask,
             save_dir="arbitrary_random_samples",
             num_samples=1)[0]

torch.jit.script(to_script)
```

```python
class GenerateModel(torch.nn.Module):
    
    def __init__(self, generators, maps, reals, amps, opt):
        super().__init__()
        self.generators = generators
        self.maps = maps
        self.reals = reals
        self.amps = amps
        self.opt = opt
        
    def forward(self, mask):
        return generate_samples(self.generators, 
             self.maps,
             self.reals,
             self.amps,
             self.opt,
             in_s=mask,
             save_dir="arbitrary_random_samples",
             num_samples=1)[0]


```

```python jupyter={"outputs_hidden": true}
model = GenerateModel(generators, use_maps, use_reals, noise_amplitudes, opt)

scripted_module = torch.jit.script(model, example_inputs={'input': mask})
```

```python jupyter={"outputs_hidden": true}

torch.onnx.export(model,               # model being run
                  mask ,                         # model input (or a tuple for multiple inputs)
                  "level_gen.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=18,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
```

```python
%%time

def get_levels(size=20):

    all_level_paths = np.random.choice(glob('apr1/lvl_*/fulldict.pth'), size=size)

    levels = []
    levnomask = []
    lis = []

    for i, path in enumerate(all_level_paths):

        try:
            levels.append(generate_level(path, maskpath='celeste/manual_level/masksmall.png')[1].cpu())
            lis.append(i)
        except Exception as e:
            print(e)
            pass

    level_paths = sorted(glob('celeste/allnopad/*.png'))
    impaths = [lpath for i,lpath in enumerate(level_paths) if i in lis]

    return levels
    
levels = get_levels()
```

## Plot all states of generation (and original level to compare)

```python
levels
```

```python
all_level_paths = sorted(glob('apr1/lvl_*/fulldict.pth'))

plot_generation(49, img_last=True, maskpath=None)
```

## Serve map using http host

```python
from flask import Flask, send_from_directory
from celeste.celeste_level.attributes import map_from_tensors
from celeste.celeste_level.level import LevelEncoder
from pathlib import Path

app = Flask(__name__)

@app.route("/")
def serve_tmp():
    
    levels = get_levels()
    
    map_ = map_from_tensors(levels)
    os.makedirs('tmp', exist_ok=True)
    LevelEncoder.write_level('tmp/ai_level.bin', map_)
    
    return send_from_directory(Path('./tmp/'), 'ai_level.bin') 

```

```python
app.run(host="0.0.0.0", port="9999")
```
