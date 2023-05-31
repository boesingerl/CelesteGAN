#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[46]:


import argparse
import base64
import gzip
import importlib.resources
import io
import json
# import cv2
import os
import sys
import tempfile
import zipfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from glob import glob
from typing import Optional


import networkx as nx
import numpy as np
import onnxruntime as ort
from joblib import Parallel, delayed

from .celeste_level.attributes import map_from_tensors
from .celeste_level.level import LevelEncoder, LevelRenderer




ARCHIVE_NAME = "celestegan.pyz"
MASK_PATH = os.path.join("data", "mask_small.json.gz")
GENERATORS_PATH = os.path.join("data", "generators", "*.json.gz")


# In[32]:


class Upscaler:
    # def dilate(img, kernel):
    #     return np.stack([cv2.dilate(x, kernel, iterations=1) for x in img])

    def __init__(self, path="upscale.onnx"):
        
        # isort: off
        sys.path += [".."]
        import celeste
        # isort: on
        
        readbytes = importlib.resources.read_binary(celeste, path)
        self.sess = ort.InferenceSession(readbytes, providers=["CPUExecutionProvider"])

        self.kernel = np.ones((3, 3)) * 1.5

    def __call__(self, image, shape):
        newimg = image.copy()
        # newimg[:, 2:] = Upscaler.dilate(newimg[0, 2:], self.kernel)[None,:,:,:]

        return self.sess.run(
            None,
            {
                "image": newimg,
                "shape0": np.array(shape[0]),
                "shape1": np.array(shape[1]),
            },
        )[0]


def format_and_use_generator(
    curr_img, G_z, count, Z_opt, pad_noise, pad_image, noise_amp, G, opt, scaler
):
    """Correctly formats input for generator and runs it through."""

    if curr_img.shape != G_z.shape:
        G_z = scaler(G_z, curr_img.shape[-2:])

    curr_img = pad_noise(curr_img)  # Curr image is z in this case
    z_add = curr_img

    G_z = pad_image(G_z)

    z_in = noise_amp * z_add + G_z

    G_z = G.run(None, {"x": z_in, "y": G_z, "temperature": np.array(1)})[0]
    return G_z


def draw_concat(
    generators,
    noise_maps,
    reals,
    noise_amplitudes,
    pad_noise,
    pad_image,
    opt,
    scaler,
    start_img=None,
    in_s=None,
    replace_scale=1,
):
    """Draw and concatenate output of the previous scale and a new noise map."""

    if start_img is None:
        G_z = np.zeros_like(noise_maps[0])
    else:
        G_z = start_img

    noise_padding = 1 * opt.num_layer
    pad_noise = pad_noise(noise_padding)
    pad_image = pad_image(noise_padding)

    for count, (G, Z_opt, real_curr, noise_amp) in enumerate(
        zip(generators, noise_maps, reals, noise_amplitudes)
    ):
        z = np.random.normal(
            size=[
                1,
                real_curr.shape[1],
                Z_opt.shape[2] - 2 * noise_padding,
                Z_opt.shape[3] - 2 * noise_padding,
            ]
        ).astype("float32")

        G_z = format_and_use_generator(
            z, G_z, count, Z_opt, pad_noise, pad_image, noise_amp, G, opt, scaler
        )

        if count == replace_scale and in_s is not None:
            in_s = scaler(in_s, G_z.shape[-2:])

            in_s[0, 20] = np.where(in_s[0, 17] > 0, 0, in_s[0, 20])

            G_z[0, 20] = in_s[0, 20]
            G_z[0, 32] = in_s[0, 32]
            G_z[0, 17] = np.where(in_s[0, 17] > 0, in_s[0, 17], G_z[0, 17])

            G_z[0, 0] = np.where(in_s[0, 20] > 0, 0, G_z[0, 0])
            G_z[0, 0] = np.where(in_s[0, 32] > 0, 0, G_z[0, 0])
            G_z[0, 1] = np.where(in_s[0, 20] > 0, 0, G_z[0, 1])
            G_z[0, 1] = np.where(in_s[0, 32] > 0, 0, G_z[0, 1])

    return G_z


class LevelGen:
    def __init__(self, generators, noise_maps, reals, noise_amplitudes):
        import torch

        self.bytes = []
        for generator, noise_map in zip(generators, noise_maps):
            bytio = io.BytesIO()

            with redirect_stdout(None):
                torch.onnx.export(
                    generator,
                    (noise_map, noise_map, 1),
                    bytio,
                    input_names=["x", "y", "temperature"],
                    output_names=["output"],
                )

            self.bytes.append(bytio)

        self.noise_amplitudes = noise_amplitudes
        self.noise_maps = noise_maps
        self.reals = reals

    def to_json(self, path):
        with gzip.open(path, "wb") as handle:
            gen_dict = {
                "gen_bytes": [
                    base64.b64encode(v.getvalue()).decode() for v in self.bytes
                ]
            }
            gen_dict["noise_maps"] = noise_maps = [
                np.asarray(nmap).tolist() for nmap in self.noise_maps
            ]
            gen_dict["noise_amps"] = [
                np.asarray(amp).tolist() for amp in self.noise_amplitudes
            ]
            gen_dict["reals"] = [np.asarray(real).tolist() for real in self.reals]

            handle.write(json.dumps(gen_dict).encode("utf-8"))

    @staticmethod
    def read_json(path):
        with gzip.open(path, "rb") as handle:
            load_dict = json.loads(handle.read().decode())

        read_gen = [
            ort.InferenceSession(
                base64.b64decode(allbytes), providers=["CPUExecutionProvider"]
            )
            for allbytes in load_dict["gen_bytes"]
        ]

        noise_maps = [np.array(x, dtype="float32") for x in load_dict["noise_maps"]]
        noise_amps = [np.array(x, dtype="float32") for x in load_dict["noise_amps"]]
        reals = [np.array(x, dtype="float32") for x in load_dict["reals"]]

        return read_gen, noise_maps, reals, noise_amps


def padding_func(padsize):
    def pad_(img):
        return np.pad(img, ((0,), (0,), (padsize,), (padsize,)))

    return pad_


@dataclass
class GenerateSamplesConfig:
    out_: Optional[str] = None  # folder containing generator files
    scale_v: float = 1.0  # vertical scale factor
    scale_h: float = 1.0  # horizontal scale factor
    gen_start_scale: int = 0  # scale to start generating in
    num_samples: int = 10  # number of samples to be generated
    save_tensors: bool = False  # save pytorch .pt tensors?
    # make 1000 samples for each mario generator specified in the code.
    make_mario_samples: bool = False
    seed_mariokart_road: bool = False  # seed mariokart generators with a road image
    # make token insert experiment (experimental!)
    token_insert_experiment: bool = False
    not_cuda: bool = False  # disables cuda
    generators_dir: Optional[str] = None
    num_layer: int = 4
    stop_scale: int = 2

    def __post_init__(self):
        self.token_list = list(range(33))


def graph_from_img(bw):
    G = nx.Graph()

    for y, row in enumerate(bw):
        for x, val in enumerate(row):
            G.add_node((x, y))

            try:
                if bw[y, x - 1].item() == val.item() and val.item() == 0:
                    if not neighbours_wall(bw, (x - 1, y)) and not neighbours_wall(
                        bw, (x, y)
                    ):
                        G.add_edge((x, y), (x - 1, y), weight=1)
                    else:
                        G.add_edge((x, y), (x - 1, y), weight=50)

                else:
                    G.add_edge((x, y), (x - 1, y), weight=100)
            except:
                pass

            try:
                if bw[y - 1, x].item() == val.item() and val.item() == 0:
                    if not neighbours_wall(bw, (x, y - 1)) and not neighbours_wall(
                        bw, (x, y)
                    ):
                        G.add_edge((x, y), (x, y - 1), weight=1)
                    else:
                        G.add_edge((x, y), (x, y - 1), weight=50)

                else:
                    G.add_edge((x, y), (x, y - 1), weight=100)
            except:
                pass

    return G


def neighbours_wall(img, point):
    x, y = point

    try:
        if img[y - 1, x] == 1:
            return True
    except:
        pass

    try:
        if img[y + 1, x] == 1:
            return True
    except:
        pass

    try:
        if img[y, x - 1] == 1:
            return True
    except:
        pass

    try:
        if img[y, x + 1] == 1:
            return True
    except:
        pass

    return False


def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def force_path(sample):
    sample = sample.copy()

    bw = sample[0].argmax(0)
    # bw[bw > 1] = 0

    G = graph_from_img(bw)

    wheres = np.where(sample[0].argmax(0) == 32)
    yfin, xfin = [
        np.sort(x)[x.shape[0] // 2] for x in np.where(sample[0].argmax(0) == 32)
    ]
    ysta, xsta = [
        np.sort(x)[x.shape[0] // 2] for x in np.where(sample[0].argmax(0) == 20)
    ]

    path = nx.astar_path(G, (xfin, yfin), (xsta, ysta), heuristic=dist, weight="weight")

    PROBLEMS = [
        "bounceBlock",
        "bigSpinner",
        "coverupWall",
        "crushBlock",
        "dashBlock",
        "fallingBlock",
        "fireBall",
        "moveBlock",
        "seeker",
        "sinkingPlatform",
        "spikesDown",
        "spikesLeft",
        "spikesRight",
        "spikesUp",
        "spinner",
        "zipMover",
    ]
    pbids = [1] + [LevelRenderer.ID_MAP[i] for i in PROBLEMS]

    def set_path(img, point):
        x, y = point

        def tryset(img, x, y):
            try:
                img[0, pbids, y, x] = 0
            except:
                pass

        tryset(img, x, y)
        tryset(img, x - 1, y)
        tryset(img, x + 1, y)
        tryset(img, x, y - 1)
        tryset(img, x, y + 1)

    for p in path:
        set_path(sample, p)

    return sample


# In[44]:
def main():
    parser = argparse.ArgumentParser(
        prog="Celeste Level Creator", description="Create celeste levels with GANs"
    )

    parser.add_argument("outpath")
    parser.add_argument("--levelsize", type=int, choices=range(1, 100), default=20)

    parsed = parser.parse_args()

    zf = zipfile.ZipFile(ARCHIVE_NAME)

    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)

        with gzip.open(os.path.join(tempdir, MASK_PATH), "rb") as handle:
            masknp = np.array(json.loads(handle.read().decode()), dtype="float32")
            masknp[0, 0] = 0
            masknp[0, 20] = masknp[0, 20] * 2
            masknp[0, 32] = masknp[0, 32] * 2

        gen_paths = np.random.choice(
            glob(os.path.join(tempdir, GENERATORS_PATH)), size=parsed.levelsize
        )
        opt = GenerateSamplesConfig()

        def level_from_path(path):
            scaler = Upscaler()

            read_gen, noise_maps, reals, noise_amps = LevelGen.read_json(path)

            level = draw_concat(
                read_gen,
                noise_maps,
                reals,
                noise_amps,
                padding_func,
                padding_func,
                opt,
                scaler,
                in_s=masknp,
            )

            scaled_mask = scaler(masknp, level.shape[-2:])
            level[0, 20] = scaled_mask[0, 20] * 20
            level[0, 32] = scaled_mask[0, 32] * 20
            level[0, 17] = scaled_mask[0, 17] * 20

            level = force_path(level)

            return level

        levels = Parallel(n_jobs=4, backend="loky")(
            delayed(level_from_path)(path) for path in gen_paths
        )

        enc_map = map_from_tensors(levels)

        LevelEncoder.write_level(
            parsed.outpath,
            enc_map,
            exec_path=os.path.join(tempdir, "celeste", "celeste_level", "json2map.rb"),
        )


if __name__ == "__main__":
    main()
