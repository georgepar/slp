import os
from collections import OrderedDict
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from loguru import logger
from pytorch_lightning.utilities import rank_zero_only
from slp.util.system import print_separator


class EarlyStoppingWithLogs(pl.callbacks.EarlyStopping):
    def _run_early_stopping_check(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Check if we should early stop on the early_stop_metric and print logs

        Args:
            trainer (pl.Trainer): Trainer
            pl_module (pl.LightningModule): Module used for training
        """
        super(EarlyStoppingWithLogs, self)._run_early_stopping_check(trainer, pl_module)
        logger.info(f"Epoch {trainer.current_epoch + 1} Early Stopping")
        print_separator(symbol="-", n=50, print_fn=logger.info)
        logger.info("{:<15} {:<15}".format("best score", self.best_score))
        logger.info(
            "{:<15} {:<15}".format("patience left", self.patience - self.wait_count)
        )

        if trainer.should_stop:  # type: ignore
            logger.info("Stopping due to early stopping")
        print_separator(symbol="#", n=50, print_fn=logger.info)


class FromLogits(pl.metrics.Metric):
    def __init__(self, metric: pl.metrics.Metric):
        """Wrap pytorch lighting metric to accept logits input

        Args:
            metric (pl.metrics.Metric): The metric to wrap, e.g. pl.metrics.Accuracy
        """
        super(FromLogits, self).__init__(
            compute_on_step=metric.compute_on_step,
            dist_sync_on_step=metric.dist_sync_on_step,
            process_group=metric.process_group,
            dist_sync_fn=metric.dist_sync_fn,
        )
        self.metric = metric

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        """Update underlying metric

        Calculate softmax under the hood and pass probs to the underlying metric

        Args:
            preds (torch.Tensor): [B, *, num_classes] Logits
            target (torch.Tensor): [B, *] Ground truths
        """
        preds = F.softmax(preds, dim=-1)
        self.metric.update(preds, target)  # type: ignore

    def compute(self) -> torch.Tensor:
        """Compute metric

        Returns:
            torch.Tensor: metric value
        """
        return self.metric.compute()  # type: ignore


class FixedWandbLogger(pl.loggers.WandbLogger):
    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: Optional[bool] = False,
        id: Optional[str] = None,
        anonymous: Optional[bool] = False,
        version: Optional[str] = None,
        project: Optional[str] = None,
        log_model: Optional[bool] = False,
        experiment: wandb.sdk.wandb_run.Run = None,
        prefix: Optional[str] = "",
        sync_step: Optional[bool] = True,
        checkpoint_dir: Optional[str] = None,
        **kwargs,
    ):
        """Wandb logger fix to save checkpoints in wandb

        Accepts an additional checkpoint_dir argument, pointing to the real checkpoint directory

        Args:
            name (Optional[str]): Display name for the run. Defaults to None.
            save_dir (Optional[str]): Path where data is saved. Defaults to None.
            offline (Optional[bool]): Run offline (data can be streamed later to wandb servers). Defaults to False.
            id (Optional[str]): Sets the version, mainly used to resume a previous run. Defaults to None.
            anonymous (Optional[bool]): Enables or explicitly disables anonymous logging. Defaults to False.
            version (Optional[str]): Sets the version, mainly used to resume a previous run. Defaults to None.
            project (Optional[str]): The name of the project to which this run will belong. Defaults to None.
            log_model (Optional[bool]): Save checkpoints in wandb dir to upload on W&B servers. Defaults to False.
            experiment ([type]): WandB experiment object. Defaults to None.
            prefix (Optional[str]): A string to put at the beginning of metric keys. Defaults to "".
            sync_step (Optional[bool]): Sync Trainer step with wandb step. Defaults to True.
            checkpoint_dir (Optional[str]): Real checkpoint dir. Defaults to None.
        """
        self._checkpoint_dir = checkpoint_dir
        super(FixedWandbLogger, self).__init__(
            name=name,
            save_dir=save_dir,
            offline=offline,
            id=id,
            anonymous=anonymous,
            version=version,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            sync_step=sync_step,
            **kwargs,
        )

    @rank_zero_only
    def finalize(self, status: str) -> None:
        """Determine where checkpoints are saved and upload to wandb servers

        Args:
            status (str): Experiment status
        """
        # offset future training logged on same W&B run

        if self._experiment is not None:
            self._step_offset = self._experiment.step

        checkpoint_dir = (
            self._checkpoint_dir if self._checkpoint_dir is not None else self.save_dir
        )

        if checkpoint_dir is None:
            logger.warning(
                "Invalid checkpoint dir. Checkpoints will not be uploaded to Wandb."
            )
            logger.info(
                "You can manually upload your checkpoints through the CLI interface."
            )

        else:
            # upload all checkpoints from saving dir

            if self._log_model:
                wandb.save(os.path.join(checkpoint_dir, "*.ckpt"))
