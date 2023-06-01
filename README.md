# CelesteGAN : A Celeste level generator using GANs !

<p align="center">
  <img src="https://github.com/boesingerl/CelesteGAN/assets/32189761/66529591-b3ee-49cd-891d-efe50f3aefe1">
</p>

## Demo

https://github.com/boesingerl/CelesteGAN/assets/32189761/52976f63-d715-433d-9d3a-df45dd40e6c1

## Getting started

This project is an Everest mod, so if you want to try it out you have to [install it first](https://everestapi.github.io/#installing-everest). Make sure you are using a (core) branch as the mod may not work on another branch.

Get the [latest release](https://github.com/boesingerl/CelesteGAN/releases) and follow the instructions on the page to setup the mod.

Then simply open the game, go into the Mod Options and generate a new gan level !

## Code

This branch contains the code for training the GANs for generating levels, the code for the Everest mod is located in the [`mod`](https://github.com/boesingerl/CelesteGAN/tree/mod) branch.

## Original work

The original code for level generation comes from TOAD-GAN (by Maren Awiszus, Frederik Schubert), which was adapted to allow level generation for the game Celeste.
See the original repo at https://github.com/Mawiszus/TOAD-GAN, and their paper:

```
@inproceedings{awiszus2020toadgan,
  title={TOAD-GAN: Coherent Style Level Generation from a Single Example},
  author={Awiszus, Maren and Schubert, Frederik and Rosenhahn, Bodo},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment},
  year={2020}
}
```


