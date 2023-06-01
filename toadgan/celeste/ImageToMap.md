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

import subprocess
import json
import tempfile
import os
import torch
from level import Level, LevelRenderer, LevelEncoder
from typing import Optional, Any, List, TypeVar, Dict
from dataclasses import dataclass, field

from attributes import map_from_tensors
import scipy
```

## Create Map from list of tensors

```python
# list of level tensors of shape (1, C, H, W)
levels = torch.load("custom_level/tmplevels.pth")

# create map from list of tensors
map_dict = map_from_tensors(levels)

# convert json to binary
LevelEncoder.write_level("custom_level/newestlevel.bin", map_dict)
```
