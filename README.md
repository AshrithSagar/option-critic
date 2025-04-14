# option-critic

![GitHub](https://img.shields.io/github/license/AshrithSagar/option-critic)
![GitHub repo size](https://img.shields.io/github/repo-size/AshrithSagar/option-critic)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

The Option-Critic Architecture

## Installation

Clone the repo and install in editable mode using `pip` from the project root directory:

```shell
pip install -e .
```

The list of dependencies are available [here](requirements.txt), which are automatically installed.

## Usage

```shell
oca run --help
```

```shell
oca run --switch-goal --env FourRooms-v0
```

```shell
oca run --switch-goal --env FourRooms-v0 --render-mode human --eval
```

```shell
oca plot --help
```

## References

- <https://github.com/lweitkamp/option-critic-pytorch>
