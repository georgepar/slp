from loguru import logger

from slp.util.log import configure_logging
from slp.config.omegaconf import OmegaConf
from slp.plbind import add_trainer_args, add_optimizer_args


def make_cli_parser(parser, datamodule_cls):
    parser = add_optimizer_args(parser)
    parser = add_trainer_args(parser)
    parser = datamodule_cls.add_argparse_args(parser)

    return parser


def parse_config(parser):
    config_file = parser.parse_args().config  # Path to config file

    # Merge Configurations Precedence: default kwarg values < default argparse values < config file values < user provided CLI args values
    if config_file is not None:
        dict_config = OmegaConf.from_yaml(config_file)
    else:
        dict_config = OmegaConf.create({})

    user_cli, default_cli = OmegaConf.from_argparse(parser)
    config = OmegaConf.merge(default_cli, dict_config, user_cli)

    # Setup logging module.
    logfile = configure_logging(f"logs/{config.trainer.experiment_name}")
    logger.info(f"Log file will be saved in {logfile}")

    logger.info("Running with the following configuration")
    logger.info(f"\n{OmegaConf.to_yaml(config)}")

    return config
