import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from argparse import Namespace
from omegaconf import DictConfig
from loguru import logger
from pytorch_lightning.core.step_result import Result

from typing import Optional, Dict, Union, List

from slp.util.types import LossType, Configuration
from slp.util.system import print_separator
from slp.config.omegaconf import OmegaConf


class _Classification(object):
    def parse_batch(self, batch):
        inputs = batch[0]
        targets = batch[1]
        return inputs, targets

    def get_predictions_and_targets(self, model, batch):
        inputs, targets = self.parse_batch(batch)
        y_pred = model(inputs)
        return y_pred, targets


class _AutoEncoder(object):
    def parse_batch(self, batch):
        inputs = batch[0]
        return inputs, inputs

    def get_predictions_and_targets(self, model, batch):
        inputs, targets = self.parse_batch(batch)
        y_pred = model(inputs)
        return y_pred.view(y_pred.size(0), -1), targets.view(targets.size(0), -1)


class _RnnClassification(object):
    def parse_batch(self, batch):
        inputs = batch[0]
        targets = batch[1]
        lengths = batch[2]
        return inputs, targets, lengths

    def get_predictions_and_targets(self, model, batch):
        inputs, targets, lengths = self.parse_batch(batch)
        y_pred = model(inputs, lengths)
        return y_pred, targets


class _TransformerClassification(object):
    def parse_batch(self, batch):
        inputs = batch[0]
        targets = batch[1]
        attention_mask = batch[2]
        return inputs, targets, attention_mask

    def get_predictions_and_targets(self, model, batch):
        inputs, targets, attention_mask = self.parse_batch(batch)
        y_pred = model(inputs, attention_mask=attention_mask)
        return y_pred, targets


class _Transformer(object):
    def parse_batch(self, batch):
        inputs = batch[0]
        targets = batch[1]
        source_mask = batch[2]
        target_mask = batch[3]
        return inputs, targets, source_mask, target_mask

    def get_predictions_and_targets(self, model, batch):
        inputs, targets, source_mask, target_mask = self.parse_batch(batch)
        y_pred = model(
            inputs, targets, source_mask=source_mask, target_mask=target_mask
        )

        y_pred = y_pred.view(-1, y_pred.size(-1))
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
    ):
        super(SimplePLModule, self).__init__()
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
            self.train_metrics = {}
            self.val_metrics = {}
            self.test_metrics = {}
        self.predictor = predictor_cls()

        if hparams is not None:
            if isinstance(hparams, Namespace):
                dict_params = vars(hparams)
            elif isinstance(hparams, DictConfig):
                dict_params = OmegaConf.to_container(hparams)
            else:
                dict_params = hparams
            self.hparams = dict_params
            self.save_hyperparameters(dict_params)

    def configure_optimizers(self):
        if self.lr_scheduler is not None:
            return self.optimizer, self.lr_scheduler
        else:
            return self.optimizer

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _compute_metrics(self, metrics, loss, y_hat, targets, mode="train"):
        def fmt(name):
            return f"{mode}_{name}"

        metrics = {fmt(k): v(y_hat, targets) for k, v in metrics.items()}
        if mode == "train":
            metrics["loss"] = loss
        else:
            metrics[fmt("loss")] = loss

        return metrics

    def log_to_console(self, metrics, mode="Training"):
        logger.info("Epoch {} {} results".format(self.current_epoch, mode))
        print_separator(symbol="-", n=50, print_fn=logger.info)
        for name, value in metrics.items():
            if name == "epoch":
                continue
            logger.info("{:<15} {:<15}".format(name, value))

        print_separator(symbol="%", n=50, print_fn=logger.info)

    def aggregate_epoch_metrics(self, outputs, mode="Training"):
        def fmt(name):
            return f"{name}" if name != "loss" else "train_loss"

        keys = list(outputs[0].keys())
        aggregated = {fmt(k): torch.stack([x[k] for x in outputs]).mean() for k in keys}
        self.log_to_console(aggregated, mode=mode)
        aggregated["epoch"] = self.current_epoch
        self.log_dict(aggregated, logger=True, prog_bar=True, on_epoch=True)

        return aggregated

    def training_step(self, batch, batch_idx):
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
        outputs = self.aggregate_epoch_metrics(outputs, mode="Training")

    def validation_step(self, batch, batch_idx):
        y_hat, targets = self.predictor.get_predictions_and_targets(self, batch)
        loss = self.criterion(y_hat, targets)
        metrics = self._compute_metrics(
            self.val_metrics, loss, y_hat, targets, mode="val"
        )

        return metrics

    def validation_epoch_end(self, outputs):
        outputs = self.aggregate_epoch_metrics(outputs, mode="Validation")

    def test_step(self, batch, batch_idx):
        y_hat, targets = self.predictor.get_predictions_and_targets(self, batch)
        loss = self.criterion(y_hat, targets)
        metrics = self._compute_metrics(
            self.test_metrics, loss, y_hat, targets, mode="test"
        )

        return metrics

    def test_epoch_end(self, outputs):
        outputs = self.aggregate_epoch_metrics(outputs, mode="Test")


def _make_specialized_pl_module(predictor_cls):
    class Module(SimplePLModule):
        def __init__(
            self,
            model: nn.Module,
            optimizer: Union[Optimizer, List[Optimizer]],
            criterion: LossType,
            lr_scheduler: Union[_LRScheduler, List[_LRScheduler]] = None,
            hparams: Configuration = None,
            metrics: Optional[Dict[str, pl.metrics.Metric]] = None,
        ):
            super(Module, self).__init__(
                model,
                optimizer,
                criterion,
                predictor_cls=predictor_cls,
                metrics=metrics,
                hparams=hparams,
            )

    return Module


PLModule = _make_specialized_pl_module(_Classification)
AutoEncoderPLModule = _make_specialized_pl_module(_AutoEncoder)
RnnPLModule = _make_specialized_pl_module(_RnnClassification)
TransformerClassificationPLModule = _make_specialized_pl_module(
    _TransformerClassification
)
TransformerPLModule = _make_specialized_pl_module(_Transformer)