import os
import numpy as np
import torch
import validators
import argparse
from torch.optim.optimizer import Optimizer
from omegaconf import DictConfig
from typing import Dict, Union, List, TypeVar, Tuple, Callable, Any

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

NdTensor = Union[np.ndarray, torch.Tensor, List[T]]
Label = Union[NdTensor, int]

Device = Union[torch.device, str]

ModuleOrOptimizer = Union[torch.nn.Module, Optimizer]

LossType = Union[torch.nn.Module, Callable]

# word2idx, idx2word, embedding vectors
Embeddings = Tuple[Dict[str, int], Dict[int, str], np.ndarray]

ValidationResult = Union[validators.ValidationFailure, bool]

GenericDict = Dict[K, V]

Configuration = Union[DictConfig, Dict[str, Any], argparse.Namespace]


def dir_path(path):
    """dir_path Type to use when parsing a path in argparse arguments


    Args:
        path (str): User provided path

    Raises:
        argparse.ArgumentTypeError: Path does not exists, so argparse fails

    Returns:
        str: User provided path

    Examples:
        >>> from slp.util.types import dir_path
        >>> import argparse
        >>> parser = argparse.ArgumentParser("My cool model")
        >>> parser.add_argument("--config", type=dir_path)
        >>> parser.parse_args(args=["--config", "my_random_config_that_does_not_exist.yaml"])
        Traceback (most recent call last):
        argparse.ArgumentTypeError: User provided path 'my_random_config_that_does_not_exist.yaml' does not exist

    """
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"User provided path '{path}' does not exist")
