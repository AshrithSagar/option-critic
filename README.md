# option-critic

![GitHub](https://img.shields.io/github/license/AshrithSagar/option-critic)
![GitHub repo size](https://img.shields.io/github/repo-size/AshrithSagar/option-critic)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Implementation of the Option-Critic Architecture | PyTorch

[DOI](https://dl.acm.org/doi/10.5555/3298483.3298491)
|
[arXiv](https://arxiv.org/abs/1609.05140)
|
[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/10916/10775)

## Installation

Clone the repo and install in editable mode using `pip` from the project root directory:

```shell
pip install -e .
```

The list of dependencies are available [here](requirements.txt), which are automatically installed.

## Usage

See all the available config options using

```shell
oca run --help
```

Check [`ConfigRunDefaults`](oca/utils/config.py) for the base defaults.
Specify any overrides using CLI arguments.

```shell
oca run --switch-goal --env FourRooms-v0 --agent OptionCritic
```

Environments (not limited to, but tested on these):

- `FourRooms-v0`
- `AsterixNoFrameskip-v4`
- `MsPacmanNoFrameskip-v4`
- `SeaquestNoFrameskip-v4`
- `ZaxxonNoFrameskip-v4`
- `CartPole-v1`

To check logs in tensorboard

```shell
tensorboard --logdir=./oca/experiments/runs/{run_name}
```

### Plots

To check available configs, use

```shell
oca plot --help
```

```shell
oca plot --run-name {run_name}
```

## References

- <https://github.com/lweitkamp/option-critic-pytorch>
- <https://pierrelucbacon.com/optioncritic-aaai2017-slides.pdf>
- <https://pierrelucbacon.com/optioncriticposter.pdf>
- <https://rll.berkeley.edu/deeprlworkshop/papers/BaconPrecup2015.pdf>
