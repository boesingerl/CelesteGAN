# Code based on https://github.com/tamarott/SinGAN
import argparse
import random
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from torch import cuda
from tap import Tap
import typing
from .utils import set_seed
from .mariokart.level_image_gen import LevelImageGen as MariokartLevelGen
from .zelda.level_image_gen import LevelImageGen as ZeldaLevelGen
from .mario.level_image_gen import LevelImageGen as MarioLevelGen
from .megaman.level_image_gen import LevelImageGen as MegamanLevelGen

import ast

class Config(Tap):
    game: Literal["mario", "mariokart", "megaman",
                  "zelda", "celeste"] = "mario"  # Which game is to be used?
    not_cuda: bool = False  # disables cuda
    netG: str = ""  # path to netG (to continue training)
    netD: str = ""  # path to netD (to continue training)
    manualSeed: Optional[int] = None
    out: str = "output"  # output directory
    input_dir: str = "input/mario"  # input directory
    input_name: str = "lvl_1-1.txt"  # input level filename
    # input level names (if multiple inputs are used)
    input_names: List[str] = ["lvl_1-1.txt", "lvl_1-2.txt"]
    # use mulitple inputs for training (use --input-names instead of --input-name)
    use_multiple_inputs: bool = False
    nfc: int = 64  # number of filters for conv layers
    ker_size: int = 3  # kernel size for conv layers
    num_layer: int = 3  # number of layers
    scales: List[float] = [0.88, 0.75, 0.5]  # Scales descending (< 1 and > 0)
    noise_update: float = 0.1  # additive noise weight
    # use reflection padding? (makes edges random)
    pad_with_noise: bool = False
    niter: int = 4000  # number of epochs to train per scale
    gamma: float = 0.1  # scheduler gamma
    lr_g: float = 0.0005  # generator learning rate
    lr_d: float = 0.0005  # discriminator learning rate
    beta1: float = 0.5  # optimizer beta
    Gsteps: int = 3  # generator inner steps
    Dsteps: int = 3  # discriminator inner steps
    lambda_grad: float = 0.1  # gradient penalty weight
    alpha: int = 100  # reconstruction loss weight
    # layer in which token groupings will be split out (<-2 means no grouping at all)
    token_insert: int = -2
    token_list: List[str] = ['!', '#', '-', '1', '@', 'C', 'S',
                             'U', 'X', 'g', 'k', 't']  # default list of 1-1

    def process_args(self):
        self.device = torch.device("cpu" if self.not_cuda else "cuda:0")
        if cuda.is_available() and self.not_cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda")

        if self.manualSeed is None:
            self.manualSeed = random.randint(1, 10000)
        set_seed(self.manualSeed)

        # Defaults for other namespace values that will be overwritten during runtime
        self.nc_current = 12  # n tokens of level 1-1
        if not hasattr(self, "out_"):
            self.out_ = "%s/%s/" % (self.out, self.input_name[:-4])
        self.outf = "0"  # changes with each scale trained
        # number of scales is implicitly defined
        self.num_scales = len(self.scales)
        self.noise_amp = 1.0  # noise amp for lowest scale always starts at 1
        self.seed_road = None  # for mario kart seed roads after training
        # which scale to stop on - usually always last scale defined
        self.stop_scale = self.num_scales
        self.ImgGen: Union[MarioLevelGen, ZeldaLevelGen,
                           MegamanLevelGen, MariokartLevelGen] = MarioLevelGen(self.game + "/sprites") if self.game in ["mario", "mariokart", "megaman",
                  "zelda"] else None

# @torch.jit.script
# class ConfigTorch:
    
#     __constants__ = ['game',
#                      'not_cuda',
#                      'netG']

#     def __init__(self):
#         self.game : str = 'celeste'
#         self.not_cuda: bool = False  # disables cuda
#         self.netG : str = ''
#         self.netD : str = ''
#         self.out = 'output'
#         self.input_dir = 'input/mario'
#         self.input_name: str = "lvl_1-1.txt"  # input level filename
#         # input level names (if multiple inputs are used)
#         self.input_names: List[str] = ["lvl_1-1.txt", "lvl_1-2.txt"]
#         # use mulitple inputs for training (use --input-names instead of --input-name)
#         self.use_multiple_inputs: bool = False
#         self.nfc: int = 64  # number of filters for conv layers
#         self.ker_size: int = 3  # kernel size for conv layers
#         self.num_layer: int = 3  # number of layers
#         self.scales: List[float] = [0.88, 0.75, 0.5]  # Scales descending (< 1 and > 0)
#         self.noise_update: float = 0.1  # additive noise weight
#         # use reflection padding? (makes edges random)
#         self.pad_with_noise: bool = False
#         self.niter: int = 4000  # number of epochs to train per scale
#         self.gamma: float = 0.1  # scheduler gamma
#         self.lr_g: float = 0.0005  # generator learning rate
#         self.lr_d: float = 0.0005  # discriminator learning rate
#         self.beta1: float = 0.5  # optimizer beta
#         self.Gsteps: int = 3  # generator inner steps
#         self.Dsteps: int = 3  # discriminator inner steps
#         self.lambda_grad: float = 0.1  # gradient penalty weight
#         self.alpha: int = 100  # reconstruction loss weight
#         # layer in which token groupings will be split out (<-2 means no grouping at all)
#         self.token_insert: int = -2
#         self.log_progress: bool = False
#         self.log_all: bool = False
#         self.token_list: List[str] = ['!', '#', '-', '1', '@', 'C', 'S',
#                                  'U', 'X', 'g', 'k', 't']  # default list of 1-1
        
#         self.device = torch.device("cpu" if self.not_cuda else "cuda:0")
#         self.manualSeed : int = 0
        
#         # set_seed(self.manualSeed)

#         # Defaults for other namespace values that will be overwritten during runtime
#         self.nc_current = 12  # n tokens of level 1-1
#         if not hasattr(self, "out_"):
#             self.out_ = "%s/%s/" % (self.out, self.input_name[:-4])
#         self.outf = "0"  # changes with each scale trained
#         # number of scales is implicitly defined
#         self.num_scales = len(self.scales)
#         self.noise_amp = 1.0  # noise amp for lowest scale always starts at 1
#         self.seed_road = None  # for mario kart seed roads after training
#         # which scale to stop on - usually always last scale defined
#         self.stop_scale = self.num_scales
#         self.ImgGen: None = None
        
#     def __getstate__(self):
#         return {
#         'game': self.game,
#         'not_cuda': self.not_cuda,
#         'netG': self.netG,
#         'netD': self.netD,
#         'out': self.out,
#         'input_dir': self.input_dir,
#         'input_name': self.input_name,
#         'input_names': self.input_names,
#         'use_multiple_inputs': self.use_multiple_inputs,
#         'nfc': self.nfc,
#         'ker_size': self.ker_size,
#         'num_layer': self.num_layer,
#         'scales': self.scales,
#         'noise_update': self.noise_update,
#         'pad_with_noise': self.pad_with_noise,
#         'niter': self.niter,
#         'gamma': self.gamma,
#         'lr_g': self.lr_g,
#         'lr_d': self.lr_d,
#         'beta1': self.beta1,
#         'Gsteps': self.Gsteps,
#         'Dsteps': self.Dsteps,
#         'lambda_grad': self.lambda_grad,
#         'alpha': self.alpha,
#         'token_insert': self.token_insert,
#         'log_progress': self.log_progress,
#         'log_all': self.log_all,
#         'token_list': self.token_list,
#         'device': self.device,
#         'manualSeed': self.manualSeed,
#         'nc_current': self.nc_current,
#         'out_': self.out_,
#         'outf': self.outf,
#         'num_scales': self.num_scales,
#         'noise_amp': self.noise_amp,
#         'seed_road': self.seed_road,
#         'stop_scale': self.stop_scale,
#         'ImgGen': self.ImgGen
#         }

#     def __setstate__(self,
#                      d # type: Dict[str, Union[str, List[str]]]
#                     ):
#         # type: (...) -> None
        
#         game: str = typing.cast(str, d['game'])# type: str
#         self.game = game
        
#         self.not_cuda = d['not_cuda'] # type: bool
#         self.netG = str(d['netG'])
#         self.netD = str(d['netD'])
#         self.out = str(d['out'])
#         self.input_dir = str(d['input_dir'])
#         self.input_name = str(d['input_name'])
#         self.input_names = list(str(d['input_names']))
#         self.use_multiple_inputs = bool(str(d['use_multiple_inputs']))
#         self.nfc = int(str(d['nfc']))
#         self.ker_size = int(str(d['ker_size']))
#         self.num_layer = d['num_layer'] # type: int
#         self.scales = d['scales'] # type: List[float]
#         self.noise_update = float(str(d['noise_update']))
#         self.pad_with_noise = d['pad_with_noise']
#         self.niter = d['niter']
#         self.gamma = d['gamma']
#         self.lr_g = d['lr_g']
#         self.lr_d = d['lr_d']
#         self.beta1 = d['beta1']
#         self.Gsteps = d['Gsteps']
#         self.Dsteps = d['Dsteps']
#         self.lambda_grad = d['lambda_grad']
#         self.alpha = d['alpha']
#         self.token_insert = d['token_insert']
#         self.log_progress = d['log_progress']
#         self.log_all = d['log_all']
#         self.token_list = d['token_list']
#         self.device = d['device']
#         self.manualSeed = d['manualSeed']
#         self.nc_current = d['nc_current']
#         self.out_ = d['out_']
#         self.outf = d['outf']
#         self.num_scales = d['num_scales']
#         self.noise_amp = d['noise_amp']
#         self.seed_road = d['seed_road']
#         self.stop_scale = d['stop_scale']
#         self.ImgGen = d['ImgGen']




