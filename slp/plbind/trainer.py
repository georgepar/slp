import os
import pytorch_lightning as pl
import wandb

from typing import Optional
from loguru import logger

from pytorch_lightning.utilities import rank_zero_only

from slp.util.system import safe_mkdirs, date_fname, has_internet_connection


class WdbLogger(pl.loggers.WandbLogger):
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
        super(WdbLogger, self).__init__(
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


def make_trainer(
    experiment_name,
    experiment_description=None,
    run_id=None,
    experiment_group=None,
    experiments_folder="experiments",
    patience=3,
    wandb_project=None,
    wandb_user=None,
    tags=None,
    swa_epoch_start=None,
    stochastic_weight_avg=False,
    auto_scale_batch_size=False,
    gpus=0,
    check_val_every_n_epoch=1,
    gradient_clip_val=0,
    precision=32,
    max_epochs=100,
    max_steps=None,
    truncated_bptt_steps=None,
    debug=False,
):
    if debug:
        trainer = pl.Trainer(overfit_batches=0.01, fast_dev_run=5)
        return trainer

    logging_dir = os.path.join(experiments_folder, experiment_name)
    safe_mkdirs(logging_dir)

    run_id = run_id if run_id is not None else date_fname()
    if run_id is None:
        run_id = date_fname()

    if run_id in os.listdir(logging_dir):
        logger.warning(
            "The run id you provided {run_id} already exists in {logging_dir}"
        )
        run_id = date_fname()
        logger.info("Setting run_id={run_id}")

    checkpoint_dir = os.path.join(logging_dir, run_id, "checkpoints")

    logger.info(f"Logs will be saved in {logging_dir}")
    logger.info(f"Logs will be saved in {checkpoint_dir}")

    if wandb_project is None:
        wandb_project = experiment_name

    connected = has_internet_connection()

    loggers = [
        pl.loggers.CSVLogger(logging_dir, name="csv_logs", version=run_id),
        WdbLogger(
            name=experiment_name,
            project=wandb_project,
            anonymous=False,
            save_dir=logging_dir,
            version=run_id,
            save_code=True,
            checkpoint_dir=checkpoint_dir,
            offline=not connected,
            log_model=True,
            entity=wandb_user,
            group=experiment_group,
            notes=experiment_description,
            tags=tags,
        ),
    ]

    logger.info(f"Configured wandb and CSV loggers.")
    logger.info(
        f"Wandb configured to run {experiment_name}/{run_id} in project {wandb_project}"
    )
    if connected:
        logger.info(f"Results will be stored online.")
    else:
        logger.info(f"Results will be stored offline due to bad internet connection.")
        logger.info(
            f"If you want to upload your results later run\n\t wandb sync {logging_dir}/wandb/run-{run_id}"
        )

    if experiment_description is not None:
        logger.info(
            f"Experiment verbose description:\n{experiment_description}\n\nTags:{'n/a' if tags is None else tags}"
        )

    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            verbose=True,
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            save_top_k=3,
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    logger.info("Configured Early stopping and Model checkpointing to track val_loss")

    trainer = pl.Trainer(
        default_root_dir=logging_dir,
        gpus=gpus,
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=loggers,
        check_val_every_n_epoch=check_val_every_n_epoch,
        gradient_clip_val=gradient_clip_val,
        auto_scale_batch_size=auto_scale_batch_size,
        stochastic_weight_avg=stochastic_weight_avg,
        precision=precision,
        truncated_bptt_steps=truncated_bptt_steps,
        terminate_on_nan=True,
    )

    return trainer


def watch_model(trainer, model):
    for log in trainer.logger.experiment:
        try:
            log.watch(model, log="all")
            logger.info("Tracking model weights & gradients in wandb.")
            break
        except:
            pass