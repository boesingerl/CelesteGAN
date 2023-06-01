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
import json
import numpy as np
import jmespath
import matplotlib.pyplot as plt
from glob import glob
import cv2
import networkx as nx
import os
import re


from celeste_level.level import Level, LevelRenderer, pad_square, LevelEncoder
from tqdm.auto import tqdm
from celeste_level.celestemap import CelesteMap
```

### Convert all levels to images

```python
def split_pad_number(s):
    match = re.match(r"([0-9]+)(.*)", s, re.I)
    if match:
        num, rest = match.groups()
        num = int(num)
        return f"{num:03d}{rest}"
    return s
```

```python
STEAM_PATH = "~/celeste_test/celeste-linux/Content/Maps/*.bin"
OUT_PATH = "../../data/input_img/"

map_paths = sorted(glob(os.path.expanduser(STEAM_PATH)))[1:2]
os.makedirs(OUT_PATH, exist_ok=True)

for map_path in tqdm(map_paths):
    # ignore mirrortemple for now because there's no straight path (teleportation triggers)
    if not "MirrorTemple" in map_path:
        lname = map_path.split("/")[-1].split(".")[0]

        zone = LevelEncoder.read_level(map_path)

        levels = jmespath.search("root.children[?name == 'levels'].children", zone)[0]
        levels = [Level(x) for x in levels]

        level_dict = {lvl.name: lvl for lvl in levels}

        cmap = CelesteMap(zone)

        path = [
            path
            for path in nx.all_shortest_paths(cmap.G, cmap.start_level, cmap.end_level)
        ][0]

        for a, b in zip(path, path[1:]):
            img = cmap.render_finish(a, b)
            reconverted = LevelRenderer.color_to_idx((img * 255).astype(int))

            colored = LevelRenderer.cm(LevelRenderer.norm(reconverted))
            plt.imsave(os.path.join(OUT_PATH, f"{lname}_{split_pad_number(a)}.png"), colored)
```
