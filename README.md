# slp

![Build Status](https://github.com/georgepar/slp/actions/workflows/ci.yml/badge.svg)
[![Maintainability](https://api.codeclimate.com/v1/badges/d3ad9729ad30aa158737/maintainability)](https://codeclimate.com/github/georgepar/slp/maintainability)


Project page: [https://github.com/georgepar/slp](https://github.com/georgepar/slp)


slp is a framework for fast and reproducible development of multimodal models, with emphasis on
NLP models.

It started as a collection of scripts and code I wrote / collected during my PhD and it evolves
accordingly.

As such, the framework is opinionated and it follows a convention over configuration approach.

A heavy emphasis is put on:  

- Enforcing best practices and reproducibility of experiments  
- Making common things fast at the top-level and not having to go through extensive configuration options  
- Remaining extendable. Extensions and modules for more use cases should be easy to add  
- Out of the box extensive logging and experiment management  
- Separating dirty / scratch code (at the script level) for quick changes and clean / polished code at the library level  

This is currently in alpha release under active development, so things may break and new features
will be added.

## Dependencies

We use [Pytorch](https://pytorch.org/) (1.7) and the following libraries

- [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/)
- [huggingface/transformers](https://huggingface.co/transformers/)
- [Wandb](https://wandb.ai/)
- Python 3.8


We use [poetry](https://python-poetry.org/) for dependency management and dependency freezing.

When you clone the repo run:

```bash
pip install poetry
poetry install
```

and a clean environment with all the dependencies will be created.
You can access it with `poetry shell`.

**Note**: Wandb logging is enabled by default. You can either

- Create an account and run `wandb login` when you clone the repo in a new machine to store the results in the online managed environment
- Run `wandb offline` when you clone the repo to disable remote sync
- Use one of their self-hosted solutions

## Documentation

Documentation can be found in [https://georgepar.github.io/slp/](https://georgepar.github.io/slp/)

## Contributions

You are welcome to open issues / PRs with improvements and bug fixes.

Since this is mostly a personal project based around workflows and practices that work for me, I don't guarantee I will accept every change, but I'm always open to discussion.

If you are going to contribute, please use the pre-commit hooks under `hooks`, otherwise the PR will not go through the CI. And never, ever touch `requirements.txt` by hand, it will automatically be exported from `poetry`

```bash

cat <<EOT >> .git/hooks/pre-commit
#!/usr/bin/env bash

bash hooks/black-check
bash hooks/export-requirements-txt
bash hooks/checks   
EOT

chmod +x .git/hooks/pre-commit
```

## Cite

If you use this code for your research, please include the following citation

```
@ONLINE {,
    author = "Georgios Paraskevopoulos",
    title  = "slp",
    year   = "2020",
    url    = "https://github.com/georgepar/slp"
}
```
