# slp

![Build Status](https://github.com/georgepar/slp/actions/workflows/ci.yml/badge.svg)
[![Maintainability](https://api.codeclimate.com/v1/badges/d3ad9729ad30aa158737/maintainability)](https://codeclimate.com/github/georgepar/slp/maintainability)


slp is a framework for fast and reproducible development of multimodal models, with emphasis on
NLP models.

It started as a collection of scripts and code I wrote / collected during my PhD and it evolves
accordingly.

As such, the framework is opinionated, meaning it follows a convention over configuration approach.
The goal is to make common patterns fast, while still remaining extendable for more complex use-cases.

This is currently in alpha release under active development, so things may break and new features
will be added.

## Tools

We use [Pytorch](https://pytorch.org/) deep learning framework and the following libraries

- [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/)
- [huggingface/transformers](https://huggingface.co/transformers/)
- [Wandb](https://wandb.ai/)

**Note**: Wandb logging is enabled by default. You can either

- Create an account and run `wandb login` when you clone the repo in a new machine to store the results in the online managed environment
- Run `wandb offline` when you clone the repo to disable remote sync
- Use one of their self-hosted solutions

## Documentation

Documentation can be found [here](https://georgepar.github.io/slp/)
