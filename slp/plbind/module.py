import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from pytorch_lightning.core.step_result import Result

from typing import Optional, Dict

from slp.util.types import LossType


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


class SimplePLModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: LossType,
        metrics: Optional[Dict[str, pl.metrics.Metric]] = None,
        predictor_cls=_Classification,
    ):
        super(SimplePLModule, self).__init__()
        self.model = model
        self.optimizer = optimizer
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

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _compute_metrics(self, metrics, loss, y_hat, targets, mode="train"):
        def fmt(name):
            return f"{mode}::{name}"

        metrics = {fmt(k): v(y_hat, targets) for k, v in metrics.items()}
        metrics[fmt("loss")] = loss

        return metrics

    def _log_iteration(self, metrics):
        def fmt(name, when):
            return f"{name}::{when}"

        for k, v in metrics.items():
            self.log(
                fmt(k, "step"),
                v,
                on_step=True,
                on_epoch=False,
                logger=True,
                prog_bar=False,
            )

            self.log(
                fmt(k, "epoch"),
                v,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

    def training_step(self, batch, batch_idx):
        y_hat, targets = self.predictor.get_predictions_and_targets(self.model, batch)
        loss = self.criterion(y_hat, targets)
        metrics = self._compute_metrics(
            self.train_metrics, loss, y_hat, targets, mode="train"
        )
        self._log_iteration(metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, targets = self.predictor.get_predictions_and_targets(self, batch)
        loss = self.criterion(y_hat, targets)
        metrics = self._compute_metrics(
            self.val_metrics, loss, y_hat, targets, mode="val"
        )
        self._log_iteration(metrics)
        metrics["loss"] = loss

        return loss

    def test_step(self, batch, batch_idx):
        y_hat, targets = self.predictor.get_predictions_and_targets(self, batch)
        loss = self.criterion(y_hat, targets)
        metrics = self._compute_metrics(
            self.test_metrics, loss, y_hat, targets, mode="test"
        )
        self._log_iteration(metrics)

        return loss


class PLModule(SimplePLModule):
    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, criterion: LossType
    ):
        super(PLModule, self).__init__(
            model, optimizer, criterion, predictor_cls=_Classification
        )


class AutoEncoderPLModule(SimplePLModule):
    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, criterion: LossType
    ):
        super(AutoEncoderPLModule, self).__init__(
            model, optimizer, criterion, predictor_cls=_AutoEncoder
        )


class RnnPLModule(SimplePLModule):
    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, criterion: LossType
    ):
        super(RnnPLModule, self).__init__(
            model, optimizer, criterion, predictor_cls=_RnnClassification
        )


class TransformerPLModule(SimplePLModule):
    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, criterion: LossType
    ):
        super(TransformerPLModule, self).__init__(
            model, optimizer, criterion, predictor_cls=_TransformerClassification
        )
