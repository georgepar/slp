# slp

<p align="center">
    <img src="https://github.com/georgepar/slp/actions/workflows/ci.yml/badge.svg" />
    <a href="https://codeclimate.com/github/georgepar/slp/maintainability" alt="Maintainability">
        <img src="https://api.codeclimate.com/v1/badges/d3ad9729ad30aa158737/maintainability" /></a>
    <a href="https://choosealicense.com/licenses/mit/" alt="License: MIT">
        <img src="https://img.shields.io/badge/license-MIT-green.svg" /></a>
    <a href="https://img.shields.io/pypi/pyversions/slp">
        <img alt="Python Version" src="https://img.shields.io/pypi/pyversions/slp" /></a>
    <a href="https://black.readthedocs.io/en/stable/" alt="Code Style: Black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
</p>

* **Repo:** [https://github.com/georgepar/slp](https://github.com/georgepar/slp)
* **Documentation:** [https://georgepar.github.io/slp/](https://georgepar.github.io/slp/)


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

## Installation

You can use slp as an external library by installing from PyPI with

```
pip install slp
```

Or you can clone it from github

```
git clone git@github.com:georgepar/slp
```

We use [poetry](https://python-poetry.org/) for dependency management

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


## Create a new project based on slp

You can use the template at [https://github.com/georgepar/cookiecutter-pytorch-slp](https://github.com/georgepar/cookiecutter-pytorch-slp)
to create a new project based on slp

```
pip install cookiecutter poetry
cookiecutter gh:georgepar/cookiecutter-pytorch-slp
# Follow the interactive configuration and a new folder with the project name you provided will appear
cd $PROJECT_NAME
poetry install  # Installs slp and all other dependencies
```

And you are good to go. Follow the instructions in the README of the new project you created. Happy coding

## Contributing

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


## Roadmap

* Optuna integration for hyperparameter tuning
* Add dataloaders for popular multimodal datasets
* Add multimodal architectures
* Add RIM, DNC and Kanerva machine implementations
* Create cookiecutter template for new project scaffolding
* Write unit tests
