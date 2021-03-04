import argparse
import os
import torch.nn as nn
import pytorch_lightning as pl

from loguru import logger
from typing import Optional, Sequence, Union

from slp.util.types import dir_path
from slp.util.system import safe_mkdirs, date_fname, has_internet_connection
from slp.plbind.helpers import FixedWandbLogger, EarlyStoppingWithLogs


def add_trainer_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Augment parser with trainer arguments

    Args:
        parent_parser (argparse.ArgumentParser): Parser created by the user

    Returns:
        argparse.ArgumentParser: Augmented parser
    """
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help="Seed for reproducibility",
    )

    parser.add_argument(
        "--config",
        dest="config",
        type=dir_path,
        default=None,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--experiment-name",
        dest="trainer.experiment_name",
        type=str,
        help="Name of the running experiment",
    )

    parser.add_argument(
        "--run-id",
        dest="trainer.run_id",
        type=str,
        help="Unique identifier for the current run. If not provided it is inferred from datetime.now()",
    )

    parser.add_argument(
        "--experiment-group",
        dest="trainer.experiment_group",
        type=str,
        help="Group of current experiment. Useful when evaluating for different seeds / cross-validation etc.",
    )

    parser.add_argument(
        "--experiments-folder",
        dest="trainer.experiments_folder",
        type=str,
        default="experiments",
        help="Top-level folder where experiment results & checkpoints are saved",
    )

    parser.add_argument(
        "--save-top-k",
        dest="trainer.save_top_k",
        type=int,
        help="Save checkpoints for top k models",
    )

    parser.add_argument(
        "--patience",
        dest="trainer.patience",
        type=int,
        help="Number of epochs to wait before early stopping",
    )

    parser.add_argument(
        "--wandb-project",
        dest="trainer.wandb_project",
        type=str,
        help="Wandb project under which results are saved",
    )

    parser.add_argument(
        "--tags",
        dest="trainer.tags",
        type=str,
        nargs="*",
        help="Tags for current run to make results searchable.",
    )

    parser.add_argument(
        "--stochastic_weight_avg",
        dest="trainer.stochastic_weight_avg",
        action="store_true",
        help="Use Stochastic weight averaging.",
    )

    parser.add_argument(
        "--gpus", dest="trainer.gpus", type=int, help="Number of GPUs to use"
    )

    parser.add_argument(
        "--val-interval",
        dest="trainer.check_val_every_n_epoch",
        type=int,
        default=1,
        help="Run validation every n epochs",
    )

    parser.add_argument(
        "--clip-grad-norm",
        dest="trainer.gradient_clip_val",
        type=float,
        help="Clip gradients with ||grad(w)|| >= args.clip_grad_norm",
    )

    parser.add_argument(
        "--epochs",
        dest="trainer.max_epochs",
        type=int,
        help="Maximum number of training epochs",
    )

    parser.add_argument(
        "--steps",
        dest="trainer.max_steps",
        type=int,
        help="Maximum number of training steps",
    )

    parser.add_argument(
        "--tbtt_steps",
        dest="trainer.truncated_bptt_steps",
        type=int,
        help="Truncated Back-propagation-through-time steps.",
    )

    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="If true, we run a full run on a small subset of the input data and overfit 10 training batches",
    )

    return parser


def add_optimizer_args(
    parent_parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Augment parser with optimizer arguments

    Args:
        parent_parser (argparse.ArgumentParser): Parser created by the user

    Returns:
        argparse.ArgumentParser: Augmented parser
    """
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument(
        "--optimizer",
        dest="optimizer",
        type=str,
        choices=[
            "Adam",
            "AdamW",
            "SGD",
            "Adadelta",
            "Adagrad",
            "Adamax",
            "ASGD",
            "RMSprop",
        ],
        default="Adam",
        help="Which optimizer to use",
    )

    parser.add_argument(
        "--lr",
        dest="optim.lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    parser.add_argument(
        "--weight-decay",
        dest="optim.weight_decay",
        type=float,
        help="Learning rate",
    )

    parser.add_argument(
        "--lr-scheduler",
        dest="lr_scheduler",
        action="store_true",
        # type=str,
        # choices=["ReduceLROnPlateau"],
        help="Use learning rate scheduling. Currently only ReduceLROnPlateau is supported out of the box",
    )

    parser.add_argument(
        "--lr-factor",
        dest="lr_schedule.factor",
        type=float,
        help="Multiplicative factor by which LR is reduced. Used if --lr-scheduler is provided.",
    )

    parser.add_argument(
        "--lr-patience",
        dest="lr_schedule.patience",
        type=int,
        help="Number of epochs with no improvement after which learning rate will be reduced. Used if --lr-scheduler is provided.",
    )

    parser.add_argument(
        "--lr-cooldown",
        dest="lr_schedule.cooldown",
        type=int,
        help="Number of epochs to wait before resuming normal operation after lr has been reduced. Used if --lr-scheduler is provided.",
    )

    parser.add_argument(
        "--min-lr",
        dest="lr_schedule.min_lr",
        type=float,
        help="Minimum lr for LR scheduling. Used if --lr-scheduler is provided.",
    )

    return parser


def make_trainer(
    experiment_name: str = "experiment",
    experiment_description: Optional[str] = None,
    run_id: Optional[str] = None,
    experiment_group: Optional[str] = None,
    experiments_folder: str = "experiments",
    save_top_k: int = 3,
    patience: int = 3,
    wandb_project: Optional[str] = None,
    wandb_user: Optional[str] = None,
    tags: Optional[Sequence] = None,
    stochastic_weight_avg: bool = False,
    auto_scale_batch_size: bool = False,
    gpus: int = 0,
    check_val_every_n_epoch: int = 1,
    gradient_clip_val: float = 0,
    precision: int = 32,
    max_epochs: Optional[int] = 100,
    max_steps: Optional[int] = None,
    truncated_bptt_steps: Optional[int] = None,
    fast_dev_run: Optional[int] = None,
    overfit_batches: Optional[int] = None,
    terminate_on_nan: bool = False,  # Be careful this makes training very slow for large models
    profiler: Optional[Union[pl.profiler.BaseProfiler, bool, str]] = "simple",
) -> pl.Trainer:
    """Configure trainer with preferred defaults

    * Experiment folder and run_id configured (based on datetime.now())
    * Wandb and CSV loggers run by default
    * Wandb configured to save code and checkpoints
    * Wandb configured in online mode except if no internet connection is available
    * Early stopping on best validation loss is configured by default
    * Checkpointing on best validation loss is configured by default
    *

    Args:
        experiment_name (str, optional): Experiment name. Defaults to "experiment".
        experiment_description (Optional[str], optional): Detailed description of the experiment. Defaults to None.
        run_id (Optional[str], optional): Unique run_id. Defaults to datetime.now(). Defaults to None.
        experiment_group (Optional[str], optional): Group experiments over multiple runs. Defaults to None.
        experiments_folder (str, optional): Folder to save outputs. Defaults to "experiments".
        save_top_k (int, optional): Save top k checkpoints. Defaults to 3.
        patience (int, optional): Patience for early stopping. Defaults to 3.
        wandb_project (Optional[str], optional): Wandb project to save the experiment. Defaults to None.
        wandb_user (Optional[str], optional): Wandb username. Defaults to None.
        tags (Optional[Sequence], optional): Additional tags to attach to the experiment. Defaults to None.
        stochastic_weight_avg (bool, optional): Use stochastic weight averaging. Defaults to False.
        auto_scale_batch_size (bool, optional): Find optimal batch size for the available resources when running
                trainer.tune(). Defaults to False.
        gpus (int, optional): number of GPUs to use. Defaults to 0.
        check_val_every_n_epoch (int, optional): Run validation every n epochs. Defaults to 1.
        gradient_clip_val (float, optional): Clip gradient norm value. Defaults to 0 (no clipping).
        precision (int, optional): Floating point precision. Defaults to 32.
        max_epochs (Optional[int], optional): Maximum number of epochs for training. Defaults to 100.
        max_steps (Optional[int], optional): Maximum number of steps for training. Defaults to None.
        truncated_bptt_steps (Optional[int], optional): Truncated back prop breaks performs backprop every k steps of much longer
                sequence. Defaults to None.
        fast_dev_run (Optional[int], optional): Run training on a small number of  batches for debugging. Defaults to None.
        overfit_batches (Optional[int], optional): Try to overfit a small number of batches for debugging. Defaults to None.
        terminate_on_nan (bool, optional): Terminate on NaN gradients. Warning this makes training slow. Defaults to False.
        profiler (Optional[Union[pl.profiler.BaseProfiler, bool, str]]): Use profiler to track execution times of each function

    Returns:
        pl.Trainer: Configured trainer
    """
    if overfit_batches is not None:
        trainer = pl.Trainer(overfit_batches=overfit_batches, gpus=gpus)
        return trainer

    if fast_dev_run is not None:
        trainer = pl.Trainer(fast_dev_run=fast_dev_run, gpus=gpus)
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
        FixedWandbLogger(
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
        EarlyStoppingWithLogs(
            monitor="val_loss",
            mode="min",
            patience=patience,
            verbose=True,
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            save_top_k=save_top_k,
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
        terminate_on_nan=terminate_on_nan,
        progress_bar_refresh_rate=10,
        profiler=profiler,
    )

    return trainer


def watch_model(trainer: pl.Trainer, model: nn.Module) -> None:
    """If wandb logger is configured track gradient and weight norms

    Args:
        trainer (pl.Trainer): Trainer
        model (nn.Module): Module to watch
    """
    if isinstance(trainer.logger.experiment, list):
        for log in trainer.logger.experiment:
            try:
                log.watch(model, log="all")
                logger.info("Tracking model weights & gradients in wandb.")
                break
            except:
                pass
    else:
        try:
            trainer.logger.experiment.watch(model, log="all")
            logger.info("Tracking model weights & gradients in wandb.")
        except:
            pass
