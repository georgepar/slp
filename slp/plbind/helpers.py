import os
import pytorch_lightning as pl
import wandb
import torch
import torch.nn.functional as F
from loguru import logger
from collections import OrderedDict
from typing import Optional
from loguru import logger

from pytorch_lightning.utilities import rank_zero_only

from slp.util.system import print_separator


class EarlyStoppingWithLogs(pl.callbacks.EarlyStopping):
    def _run_early_stopping_check(self, trainer, pl_module):
        super(EarlyStoppingWithLogs, self)._run_early_stopping_check(trainer, pl_module)
        logger.info(f"Epoch {trainer.current_epoch} Early Stopping")
        print_separator(symbol="-", n=50, print_fn=logger.info)
        logger.info("{:<15} {:<15}".format("best score", self.best_score))
        logger.info(
            "{:<15} {:<15}".format("patience left", self.patience - self.wait_count)
        )
        if trainer.should_stop:
            logger.info("Stopping due to early stopping")
        print_separator(symbol="#", n=50, print_fn=logger.info)


class FromLogits(pl.metrics.Metric):
    def __init__(self, metric):
        super(FromLogits, self).__init__(
            compute_on_step=metric.compute_on_step,
            dist_sync_on_step=metric.dist_sync_on_step,
            process_group=metric.process_group,
            dist_sync_fn=metric.dist_sync_fn,
        )
        self.metric = metric

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = F.softmax(preds, dim=-1)
        self.metric.update(preds, target)

    def compute(self) -> torch.Tensor:
        return self.metric.compute()


class Perplexity(pl.metrics.Metric):
    def __init__(
        self,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None
    ):
        super(Perplexity, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("avg_xentropy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
 
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.avg_xentropy += F.cross_entropy(preds, target)
        self.total += 1

    def compute(self) -> torch.Tensor:
        avg_xentropy = self.avg_xentropy / self.total
        ppl = torch.exp(avg_xentropy)
        return ppl


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
        experiment=None,
        prefix: Optional[str] = "",
        sync_step: Optional[bool] = True,
        checkpoint_dir: Optional[str] = None,
        **kwargs,
    ):
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
        # offset future training logged on same W&B run
        if self._experiment is not None:
            self._step_offset = self._experiment.step

        checkpoint_dir = (
            self._checkpoint_dir if self._checkpoint_dir is not None else self.save_dir
        )

        # upload all checkpoints from saving dir
        if self._log_model:
            wandb.save(os.path.join(checkpoint_dir, "*.ckpt"))
