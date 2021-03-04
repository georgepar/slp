from abc import ABC, abstractmethod

from argparse import Namespace
from typing import Any, Dict, List, Optional, Union, cast, Tuple
from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning.core.step_result import Result
from slp.config.omegaconf import OmegaConf
from slp.util.pytorch import pad_mask, subsequent_mask
from slp.util.system import print_separator
from slp.util.types import Configuration, LossType
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class _Predictor(ABC):
    """Base predictor class

    Define an interface that can be used to extend the lightning module to new tasks and models

    * parse_batch: Parse input batch and extract necessery masks etc.
    * get_predictions_and_targets: Perform a forward pass through the model to get the logits and return logits and targets
    """

    @abstractmethod
    def parse_batch(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Abstract parse_batch method to be implemented by child class

        Args:
            batch (Tuple[torch.Tensor, ...]): A tuple of tensors that contains inputs to the model and targets

        Returns:
            Tuple[torch.Tensor, ...]: The processed inputs, ready to provide to the model
        """
        pass

    @abstractmethod
    def get_predictions_and_targets(
        self, model: nn.Module, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Abstract get_predictions and targets method to be implemented by child class

        This method gets exposed to the PLModule classes

        Args:
            model (nn.Module): model to use for forward pass
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple of tensors that contains inputs to the model and targets

        **Note**: Maybe it should be useful to move loss calculation here. Then multitask learning and auxiliary losses should be easier

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (logits, ground_truths), ready to be passed to the loss function
        """
        pass


class _Classification(_Predictor):
    """Classification task"""

    def parse_batch(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Parse incoming batch

        Input batch just contains inputs and targets

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): (inputs, labels)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (inputs, labels)
        """
        inputs = batch[0]
        targets = batch[1]

        return inputs, targets

    def get_predictions_and_targets(
        self, model: nn.Module, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and ground truths to be passed in loss function

        Args:
            model (nn.Module): Model to use for prediction
            batch (Tuple[torch.Tensor, torch.Tensor]): (inputs, labels)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (logits, labels)
        """
        inputs, targets = self.parse_batch(batch)
        y_pred = model(inputs)

        return y_pred, targets


class _AutoEncoder(_Predictor):
    """Autoencoder task"""

    def parse_batch(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Parse incoming batch

        Input batch just contains inputs. Targets are the same as inputs, because we are doing reconstruction.

        Args:
            batch (Tuple[torch.Tensor]): (inputs)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (inputs, inputs)
        """
        inputs = batch[0]

        return inputs, inputs

    def get_predictions_and_targets(
        self, model: nn.Module, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and ground truths to be passed in loss function

        Args:
            model (nn.Module): Model to use for prediction
            batch (Tuple[torch.Tensor, torch.Tensor]): (inputs, inputs)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (logits, inputs)
        """
        inputs, targets = self.parse_batch(batch)
        y_pred = model(inputs)

        return y_pred.view(y_pred.size(0), -1), targets.view(targets.size(0), -1)


class _RnnClassification(_Predictor):
    """RNN classification task"""

    def parse_batch(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Parse incoming batch

        Input batch just contains inputs, targets and lengths.
        Comes from slp.data.collators.SequentialCollator.

        Args:
            batch (Tuple[torch.Tensor]): (inputs)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (inputs, inputs)
        """
        inputs = batch[0]
        targets = batch[1]
        lengths = batch[2]

        return inputs, targets, lengths

    def get_predictions_and_targets(
        self, model: nn.Module, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and ground truths to be passed in loss function

        Args:
            model (nn.Module): Model to use for prediction
            batch (Tuple[torch.Tensor, torch.Tensor]): (inputs, inputs)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (logits, inputs)
        """
        inputs, targets, lengths = self.parse_batch(batch)
        y_pred = model(inputs, lengths)

        return y_pred, targets


class _TransformerClassification(_Predictor):
    """Transformer classification task"""

    def parse_batch(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Parse incoming batch

        Input batch just contains inputs, targets and lengths.
        Comes from slp.data.collators.SequentialCollator.
        Create pad masks to be passed to transformer attention

        Args:
            batch (Tuple[torch.Tensor]): (inputs)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (inputs, inputs)
        """
        inputs = batch[0]
        targets = batch[1]
        lengths = batch[2]
        attention_mask = pad_mask(lengths)

        return inputs, targets, attention_mask

    def get_predictions_and_targets(
        self, model: nn.Module, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and ground truths to be passed in loss function

        Args:
            model (nn.Module): Model to use for prediction
            batch (Tuple[torch.Tensor, torch.Tensor]): (inputs, inputs)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (logits, inputs)
        """
        inputs, targets, attention_mask = self.parse_batch(batch)
        y_pred = model(inputs, attention_mask=attention_mask)

        return y_pred, targets


class _Transformer(_Predictor):
    """Generic transformer seq2seq task"""

    def parse_batch(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Parse incoming batch

        Input batch just contains inputs, targets and lengths.
        Comes from slp.data.collators.SequentialCollator.
        Create pad masks and subsequent_masks to be passed to transformer attention

        Args:
            batch (Tuple[torch.Tensor]): (inputs)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (inputs, inputs)
        """
        inputs = batch[0]
        targets = batch[1]
        lengths_inputs = batch[2]
        lengths_targets = batch[3]

        max_length_inputs = torch.max(lengths_inputs)
        max_length_targets = torch.max(lengths_targets)

        pad_inputs = pad_mask(
            lengths_inputs,
            max_length=max_length_inputs,
        ).unsqueeze(-2)
        pad_targets = pad_mask(
            lengths_targets,
            max_length=max_length_targets,
        ).unsqueeze(-2)
        sub_m = subsequent_mask(max_length_targets)  # type: ignore
        pad_targets = pad_targets * sub_m.to(pad_targets.device)

        return inputs, targets, pad_inputs, pad_targets

    def get_predictions_and_targets(
        self, model: nn.Module, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and ground truths to be passed in loss function

        Args:
            model (nn.Module): Model to use for prediction
            batch (Tuple[torch.Tensor, torch.Tensor]): (inputs, inputs)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (logits, inputs)
        """
        inputs, targets, source_mask, target_mask = self.parse_batch(batch)
        y_pred = model(
            inputs, targets, source_mask=source_mask, target_mask=target_mask
        )

        y_pred = y_pred.view(-1, y_pred.size(-1))
        targets = targets.view(-1)

        return y_pred, targets


class _BertSequenceClassification(_Predictor):
    """ Bert Classification task"""

    def parse_batch(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Parse incoming batch

        Input batch just contains inputs, targets and lengths.
        Comes from slp.data.collators.SequentialCollator.
        Create pad masks to be passed to BERT attention

        Args:
            batch (Tuple[torch.Tensor]): (inputs)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (inputs, inputs)
        """
        inputs = batch[0]
        targets = batch[1]
        lengths = batch[2]

        attention_mask = pad_mask(lengths)

        return inputs, targets, attention_mask

    def get_predictions_and_targets(
        self, model: nn.Module, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and ground truths to be passed in loss function

        Args:
            model (nn.Module): Model to use for prediction
            batch (Tuple[torch.Tensor, torch.Tensor]): (inputs, inputs)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (logits, inputs)
        """
        inputs, targets, attention_mask = self.parse_batch(batch)
        out = model(
            input_ids=inputs,
            attention_mask=attention_mask,
            labels=None,
            return_dict=False,
        )
        y_pred = out[0].view(-1, out[0].size(-1))
        targets = targets.view(-1)

        return y_pred, targets


class SimplePLModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Union[Optimizer, List[Optimizer]],
        criterion: LossType,
        lr_scheduler: Union[_LRScheduler, List[_LRScheduler]] = None,
        hparams: Configuration = None,
        metrics: Optional[Dict[str, pl.metrics.Metric]] = None,
        predictor_cls=_Classification,
        calculate_perplexity: bool = False,  # for LM. Dirty but much more efficient
    ):
        """LightningModule wrapper for a (model, optimizer, criterion, lr_scheduler) tuple

        Handles the boilerplate for metrics calculation and logging and defines the train_step / val_step / test_step
        with use of the predictor helper classes (e.g. _Classification, _RnnClassification)

        Args:
            model (nn.Module): Module to use for prediction
            optimizer (Union[Optimizer, List[Optimizer]]): Optimizers to use for training
            criterion (LossType): Task loss
            lr_scheduler (Union[_LRScheduler, List[_LRScheduler]], optional): Learning rate scheduler. Defaults to None.
            hparams (Configuration, optional): Hyperparameter values. This ensures they are logged with trainer.loggers. Defaults to None.
            metrics (Optional[Dict[str, pl.metrics.Metric]], optional): Metrics to track. Defaults to None.
            predictor_cls ([type], optional): Class that defines a parse_batch and a
                    get_predictions_and_targets method. Defaults to _Classification.
            calculate_perplexity (bool, optional): Whether to calculate perplexity.
                    Would be cleaner as a metric, but this is more efficient. Defaults to False.
        """
        super(SimplePLModule, self).__init__()
        self.calculate_perplexity = calculate_perplexity
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion

        if metrics is not None:
            self.train_metrics = nn.ModuleDict(metrics)
            self.val_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
            self.test_metrics = nn.ModuleDict(
                {k: v.clone() for k, v in metrics.items()}
            )
        else:
            self.train_metrics = nn.ModuleDict(modules=None)
            self.val_metrics = nn.ModuleDict(modules=None)
            self.test_metrics = nn.ModuleDict(modules=None)
        self.predictor = predictor_cls()

        if hparams is not None:
            if isinstance(hparams, Namespace):
                dict_params = vars(hparams)
            elif isinstance(hparams, DictConfig):
                dict_params = cast(Dict[str, Any], OmegaConf.to_container(hparams))
            else:
                dict_params = hparams
            # self.hparams = dict_params
            self.save_hyperparameters(dict_params)

    def configure_optimizers(self):
        """Return optimizers and learning rate schedulers

        Returns:
            Tuple[List[Optimizer], List[_LRScheduler]]: (optimizers, lr_schedulers)
        """
        if self.lr_scheduler is not None:
            return self.optimizer, self.lr_scheduler
        else:
            return self.optimizer

    def forward(self, *args, **kwargs):
        """ Call wrapped module forward"""
        return self.model(*args, **kwargs)

    def _compute_metrics(self, metrics, loss, y_hat, targets, mode="train"):
        """Compute all metrics and aggregate in a dict

        Args:
            metrics (Dict[str, pl.metrics.Metric]): metrics to compute
            loss (torch.Tensor): Computed loss
            y_hat (torch.Tensor): Logits
            targets (torch.Tensor): Ground Truths
            mode (str, optional): "train", "val" or "test". Defaults to "train".
        """

        def fmt(name):
            """Format metric name"""
            return f"{mode}_{name}"

        metrics = {f"{mode}_{k}": v(y_hat, targets) for k, v in metrics.items()}

        if mode == "train":
            metrics["loss"] = loss
        else:
            metrics[fmt("loss")] = loss

        if self.calculate_perplexity:
            metrics[fmt("ppl")] = torch.exp(loss)

        return metrics

    def log_to_console(self, metrics, mode="Training"):
        """Log metrics to console

        Args:
            metrics (Dict[str, torch.Tensor]): Computed metrics
            mode (str, optional): "Training", "Validation" or "Testing". Defaults to "Training".
        """
        logger.info("Epoch {} {} results".format(self.current_epoch + 1, mode))
        print_separator(symbol="-", n=50, print_fn=logger.info)

        for name, value in metrics.items():
            if name == "epoch":
                continue
            logger.info("{:<15} {:<15}".format(name, value))

        print_separator(symbol="%", n=50, print_fn=logger.info)

    def aggregate_epoch_metrics(self, outputs, mode="Training"):
        """Aggregate metrics over a whole epoch

        Args:
            outputs (List[Dict[str, torch.Tensor]]): Aggregated outputs from train_step, validation_step or test_step
            mode (str, optional): "Training", "Validation" or "Testing". Defaults to "Training".
        """

        def fmt(name):
            """Format metric name"""
            return f"{name}" if name != "loss" else "train_loss"

        keys = list(outputs[0].keys())
        aggregated = {fmt(k): torch.stack([x[k] for x in outputs]).mean() for k in keys}
        self.log_to_console(aggregated, mode=mode)
        aggregated["epoch"] = self.current_epoch + 1
        self.log_dict(aggregated, logger=True, prog_bar=True, on_epoch=True)

        return aggregated

    def training_step(self, batch, batch_idx):
        """Compute loss for a single training step and log metrics to loggers

        Args:
            batch (Tuple[torch.Tensor, ...]): Input batch
            batch_idx (int): Index of batch

        Returns:
            Dict[str, torch.Tensor]: computed metrics
        """
        y_hat, targets = self.predictor.get_predictions_and_targets(self.model, batch)
        loss = self.criterion(y_hat, targets)
        metrics = self._compute_metrics(
            self.train_metrics, loss, y_hat, targets, mode="train"
        )

        self.log_dict(
            {k: v for k, v in metrics.items()},
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=False,
        )

        metrics["loss"] = loss

        return metrics

    def training_epoch_end(self, outputs):
        """Aggregate metrics of a training epoch

        Args:
            outputs (List[Dict[str, torch.Tensor]]): Aggregated outputs from train_step
        """
        outputs = self.aggregate_epoch_metrics(outputs, mode="Training")

    def validation_step(self, batch, batch_idx):
        """Compute loss for a single validation step and log metrics to loggers

        Args:
            batch (Tuple[torch.Tensor, ...]): Input batch
            batch_idx (int): Index of batch

        Returns:
            Dict[str, torch.Tensor]: computed metrics
        """
        y_hat, targets = self.predictor.get_predictions_and_targets(self, batch)
        loss = self.criterion(y_hat, targets)
        metrics = self._compute_metrics(
            self.val_metrics, loss, y_hat, targets, mode="val"
        )

        return metrics

    def validation_epoch_end(self, outputs):
        """Aggregate metrics of a validation epoch

        Args:
            outputs (List[Dict[str, torch.Tensor]]): Aggregated outputs from validation_step
        """
        outputs = self.aggregate_epoch_metrics(outputs, mode="Validation")

    def test_step(self, batch, batch_idx):
        """Compute loss for a single test step and log metrics to loggers

        Args:
            batch (Tuple[torch.Tensor, ...]): Input batch
            batch_idx (int): Index of batch

        Returns:
            Dict[str, torch.Tensor]: computed metrics
        """
        y_hat, targets = self.predictor.get_predictions_and_targets(self, batch)
        loss = self.criterion(y_hat, targets)
        metrics = self._compute_metrics(
            self.test_metrics, loss, y_hat, targets, mode="test"
        )

        return metrics

    def test_epoch_end(self, outputs):
        """Aggregate metrics of a test epoch

        Args:
            outputs (List[Dict[str, torch.Tensor]]): Aggregated outputs from test_step
        """
        outputs = self.aggregate_epoch_metrics(outputs, mode="Test")


def _make_specialized_pl_module(predictor_cls):
    """Create a LightningModule wrapper using the provided predictor class

    Args:
        predictor_cls: Class that defines parse_batch and get_predictions_and_targets

    Returns:
        pl.LightningModule: Configured LightningModule
    """

    class Module(SimplePLModule):
        def __init__(
            self,
            model: nn.Module,
            optimizer: Union[Optimizer, List[Optimizer]],
            criterion: LossType,
            lr_scheduler: Union[_LRScheduler, List[_LRScheduler]] = None,
            hparams: Configuration = None,
            metrics: Optional[Dict[str, pl.metrics.Metric]] = None,
            calculate_perplexity=False,
        ):
            """Pass arguments through to base class"""
            super(Module, self).__init__(
                model,
                optimizer,
                criterion,
                predictor_cls=predictor_cls,
                lr_scheduler=lr_scheduler,
                hparams=hparams,
                metrics=metrics,
                calculate_perplexity=calculate_perplexity,
            )

    return Module


PLModule = _make_specialized_pl_module(_Classification)
AutoEncoderPLModule = _make_specialized_pl_module(_AutoEncoder)
RnnPLModule = _make_specialized_pl_module(_RnnClassification)
TransformerClassificationPLModule = _make_specialized_pl_module(
    _TransformerClassification
)
TransformerPLModule = _make_specialized_pl_module(_Transformer)
BertPLModule = _make_specialized_pl_module(_BertSequenceClassification)
