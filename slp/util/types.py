import numpy as np
import torch
import validators
from torch.optim.optimizer import Optimizer
from omegaconf import DictConfig
from argparse import Namespace
from typing import Dict, Union, List, TypeVar, Tuple, Callable, Any

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

NdTensor = Union[np.ndarray, torch.Tensor, List[T]]

Device = Union[torch.device, str]

ModuleOrOptimizer = Union[torch.nn.Module, Optimizer]

LossType = Union[torch.nn.Module, Callable]

# word2idx, idx2word, embedding vectors
Embeddings = Tuple[Dict[str, int], Dict[int, str], np.ndarray]

ValidationResult = Union[validators.ValidationFailure, bool]

GenericDict = Dict[K, V]

Configuration = Union[DictConfig, Dict[str, Any], Namespace]