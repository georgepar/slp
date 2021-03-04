import argparse

import pytorch_lightning as pl

from loguru import logger

from typing import IO, Union, Optional, List

from slp.config.omegaconf import OmegaConfExtended as OmegaConf
from slp.plbind import add_trainer_args, add_optimizer_args


def make_cli_parser(
    parser: argparse.ArgumentParser, datamodule_cls: pl.LightningDataModule
):
    """make_cli_parser Augment an argument parser for slp with the default arguments

    Default arguments for training, logging, optimization etc. are added to the input {parser}.
    If you use make_cli_parser, the following command line arguments will be included

        usage: my_script.py [-h] [--optimizer {Adam,AdamW,SGD,Adadelta,Adagrad,Adamax,ASGD,RMSprop}] [--lr OPTIM.LR] [--weight-decay OPTIM.WEIGHT_DECAY]
                        [--lr-scheduler] [--lr-factor LR_SCHEDULE.FACTOR] [--lr-patience LR_SCHEDULE.PATIENCE] [--lr-cooldown LR_SCHEDULE.COOLDOWN] [--min-lr LR_SCHEDULE.MIN_LR] [--seed SEED]
                        [--config CONFIG] [--experiment-name TRAINER.EXPERIMENT_NAME] [--run-id TRAINER.RUN_ID] [--experiment-group TRAINER.EXPERIMENT_GROUP]
                        [--experiments-folder TRAINER.EXPERIMENTS_FOLDER] [--save-top-k TRAINER.SAVE_TOP_K] [--patience TRAINER.PATIENCE] [--wandb-project TRAINER.WANDB_PROJECT]
                        [--tags [TRAINER.TAGS [TRAINER.TAGS ...]]] [--stochastic_weight_avg] [--gpus TRAINER.GPUS] [--val-interval TRAINER.CHECK_VAL_EVERY_N_EPOCH]
                        [--clip-grad-norm TRAINER.GRADIENT_CLIP_VAL] [--epochs TRAINER.MAX_EPOCHS] [--steps TRAINER.MAX_STEPS] [--tbtt_steps TRAINER.TRUNCATED_BPTT_STEPS] [--debug]
                        [--val-percent DATA.VAL_PERCENT] [--test-percent DATA.TEST_PERCENT] [--bsz DATA.BATCH_SIZE] [--bsz-eval DATA.BATCH_SIZE_EVAL] [--num-workers DATA.NUM_WORKERS]
                        [--pin-memory] [--drop-last] [--shuffle-eval]

        optional arguments:
        -h, --help            show this help message and exit
        --optimizer {Adam,AdamW,SGD,Adadelta,Adagrad,Adamax,ASGD,RMSprop}
                                Which optimizer to use
        --lr OPTIM.LR         Learning rate
        --weight-decay OPTIM.WEIGHT_DECAY
                                Learning rate
        --lr-scheduler        Use learning rate scheduling. Currently only ReduceLROnPlateau is supported out of the box
        --lr-factor LR_SCHEDULE.FACTOR
                                Multiplicative factor by which LR is reduced. Used if --lr-scheduler is provided.
        --lr-patience LR_SCHEDULE.PATIENCE
                                Number of epochs with no improvement after which learning rate will be reduced. Used if --lr-scheduler is provided.
        --lr-cooldown LR_SCHEDULE.COOLDOWN
                                Number of epochs to wait before resuming normal operation after lr has been reduced. Used if --lr-scheduler is provided.
        --min-lr LR_SCHEDULE.MIN_LR
                                Minimum lr for LR scheduling. Used if --lr-scheduler is provided.
        --seed SEED           Seed for reproducibility
        --config CONFIG       Path to YAML configuration file
        --experiment-name TRAINER.EXPERIMENT_NAME
                                Name of the running experiment
        --run-id TRAINER.RUN_ID
                                Unique identifier for the current run. If not provided it is inferred from datetime.now()
        --experiment-group TRAINER.EXPERIMENT_GROUP
                                Group of current experiment. Useful when evaluating for different seeds / cross-validation etc.
        --experiments-folder TRAINER.EXPERIMENTS_FOLDER
                                Top-level folder where experiment results & checkpoints are saved
        --save-top-k TRAINER.SAVE_TOP_K
                                Save checkpoints for top k models
        --patience TRAINER.PATIENCE
                                Number of epochs to wait before early stopping
        --wandb-project TRAINER.WANDB_PROJECT
                                Wandb project under which results are saved
        --tags [TRAINER.TAGS [TRAINER.TAGS ...]]
                                Tags for current run to make results searchable.
        --stochastic_weight_avg
                                Use Stochastic weight averaging.
        --gpus TRAINER.GPUS   Number of GPUs to use
        --val-interval TRAINER.CHECK_VAL_EVERY_N_EPOCH
                                Run validation every n epochs
        --clip-grad-norm TRAINER.GRADIENT_CLIP_VAL
                                Clip gradients with ||grad(w)|| >= args.clip_grad_norm
        --epochs TRAINER.MAX_EPOCHS
                                Maximum number of training epochs
        --steps TRAINER.MAX_STEPS
                                Maximum number of training steps
        --tbtt_steps TRAINER.TRUNCATED_BPTT_STEPS
                                Truncated Back-propagation-through-time steps.
        --debug               If true, we run a full run on a small subset of the input data and overfit 10 training batches
        --val-percent DATA.VAL_PERCENT
                                Percent of validation data to be randomly split from the training set, if no validation set is provided
        --test-percent DATA.TEST_PERCENT
                                Percent of test data to be randomly split from the training set, if no test set is provided
        --bsz DATA.BATCH_SIZE
                                Training batch size
        --bsz-eval DATA.BATCH_SIZE_EVAL
                                Evaluation batch size
        --num-workers DATA.NUM_WORKERS
                                Number of workers to be used in the DataLoader
        --pin-memory          Pin data to GPU memory for faster data loading
        --drop-last           Drop last incomplete batch
        --shuffle-eval        Shuffle val & test sets
    Args:
        parser (argparse.ArgumentParser): A parent argument to be augmented
        datamodule_cls (pytorch_lightning.LightningDataModule): A data module class that injects arguments through the add_argparse_args method

    Returns:
        argparse.ArgumentParser: The augmented command line parser

    Examples:
        >>> import argparse
        >>> from slp.plbind.dm import PLDataModuleFromDatasets
        >>> parser = argparse.ArgumentParser("My cool model")
        >>> parser.add_argument("--hidden", dest="model.hidden", type=int)  # Create parser with model arguments and anything else you need
        >>> parser = make_cli_parser(parser, PLDataModuleFromDatasets)
        >>> args = parser.parse_args(args=["--bsz", "64", "--lr", "0.01"])
        >>> args.data.batch_size
        64
        >>> args.optim.lr
        0.01
    """
    parser = add_optimizer_args(parser)
    parser = add_trainer_args(parser)
    parser = datamodule_cls.add_argparse_args(parser)

    return parser


def parse_config(
    parser: argparse.ArgumentParser,
    config_file: Union[str, IO],
    args: Optional[List[str]] = None,
):
    """parse_config Parse a provided YAML config file and command line args and merge them

    During experimentation we want ideally to have a configuration file with the model and training configuration,
    but also be able to run quick experiments using command line args.
    This function allows you to double dip, by overriding values in a YAML config file through user provided command line arguments.

    The precedence for merging is as follows
       * default cli args values < config file values < user provided cli args

    E.g.:

       * if you don't include a value in your configuration it will take the default value from the argparse arguments
       * if you provide a cli arg (e.g. run the script with --bsz 64) it will override the value in the config file

    Note we use an extended OmegaConf istance to achieve this (see slp.config.omegaconf.OmegaConf)

    Args:
        parser (argparse.ArgumentParser): The argument parser you want to use
        config_file (Union[str, IO]): Configuration file name or file descriptor
        args (Optional[List[str]]): Optional input sys.argv style args. Useful for testing.
            Use this only for testing. By default it uses sys.argv[1:]

    Returns:
        OmegaConf.DictConfig: The parsed configuration as an OmegaConf DictConfig object

    Examples:
        >>> import io
        >>> from slp.config.config_parser import parse_config
        >>> mock_config_file = io.StringIO('''
        model:
          hidden: 100
        ''')
        >>> parser = argparse.ArgumentParser("My cool model")
        >>> parser.add_argument("--hidden", dest="model.hidden", type=int, default=20)
        >>> cfg = parse_config(parser, mock_config_file)
        {'model': {'hidden': 100}}
        >>> type(cfg)
        <class 'omegaconf.dictconfig.DictConfig'>
        >>> cfg = parse_config(parser, mock_config_file, args=["--hidden", "200"])
        {'model': {'hidden': 200}}
        >>> mock_config_file = io.StringIO('''
        random_value: hello
        ''')
        >>> cfg = parse_config(parser, mock_config_file)
        {'model': {'hidden': 20}, 'random_value': 'hello'}
    """
    # Merge Configurations Precedence: default kwarg values < default argparse values < config file values < user provided CLI args values
    if config_file is not None:
        dict_config = OmegaConf.from_yaml(config_file)  # type: ignore
    else:
        dict_config = OmegaConf.create({})

    user_cli, default_cli = OmegaConf.from_argparse(parser)
    config = OmegaConf.merge(default_cli, dict_config, user_cli)

    logger.info("Running with the following configuration")
    logger.info(f"\n{OmegaConf.to_yaml(config)}")

    return config
