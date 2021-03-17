import math
from typing import Any, Callable, Dict, Union, cast

from loguru import logger
from omegaconf import DictConfig
from ray import tune
from ray.tune.integration.wandb import WandbLogger
from ray.tune.suggest.optuna import OptunaSearch

from slp.config.omegaconf import OmegaConfExtended as OmegaConf
from slp.util.system import date_fname, has_internet_connection, yaml_dump


def _extract_wandb_config(config: DictConfig) -> DictConfig:
    """Copy wandb configuration from trainer subdict to wandb subdict
    Ray tune requires a separate wandb configuration entry in the input config dict
    Our configuration organizes these parameters under trainer e.g.:

    > trainer:
    >   wandb_project: ...
    >   wandb_user: ...

    This function takes all relevant wandb configuration options and puts them under a wandb
    entry, converting them in a form that is compatible with wandb.init(), e.g.:

    > wandb:
    >   project: trainer_config["wandb_project"]
    >   entity: trainer_config["wandb_user"]

    This function is implicitly called by run_tuning and you should not have to directly interact
    with it.

    Args:
        config (omegaconf.DictConfig): The parsed configuration

    Returns:
        (omegaconf.DictConfig): The converted configuration
    """
    cfg = cast(DictConfig, OmegaConf.to_container(config))
    train_cfg = cfg["trainer"]
    experiment_name = train_cfg.get("experiment_name", "experiment")
    connected = has_internet_connection()
    offline_run = train_cfg.get("force_wandb_offline", False) or not connected

    cfg["wandb"] = {
        "name": experiment_name,
        "entity": train_cfg.get("wandb_user", None),
        "project": train_cfg.get("wandb_project", None),
        "tags": train_cfg.get("tags", []),
        "group": train_cfg.get("experiment_group", f"tune-{date_fname()}"),
        "save_code": True,
        "notes": train_cfg.get(
            "experiment_description", f"Tuning for {experiment_name}"
        ),
        "mode": "offline" if offline_run else "online",
        "anonymous": "never",
    }

    config = OmegaConf.create(cfg)

    return config


def run_tuning(
    config: DictConfig,
    output_config_file: str,
    train_fn: Callable[[Dict[str, Any], Any, Any], None],
    config_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    train: Any = None,
    val: Any = None,
):
    """Run distributed hyperparameter tuning using ray tune

    Uses Optuna TPE search algorithm and ASHA pruning strategy

    Args:
        config (omegaconf.DictConfig): The parsed configuration
        output_config_file (str): Path to save the optimal configuration that yields the best
            result
        train_fn (Callable[[Dict[str, Any], Any, Any], None]): Train function that takes the
            configuration as a python dict, train dataset and validation dataset and fits the
            model. This function is used to create the trainable that will run when calling
            ray.tune.run
        config_fn (Callable[[Dict[str, Any]], Dict[str, Any]]): Configuration function that
            constructs the search space by overriding entries in the input configuration
        train (Dataset): Torch dataset or corpus that will be used for training
        val (Dataset): Torch dataset or corpus that will be used for validation

    Returns:
        Dict[str, Any]: The configuration for the best trial

    Examples:
        >>> # Make search space
        >>> def configure_search_space(config):
        >>>     config["optimizer"] = tune.choice(["SGD", "Adam", "AdamW"])
        >>>     config["optim"]["lr"] = tune.loguniform(1e-4, 1e-1)
        >>>     config["optim"]["weight_decay"] = tune.loguniform(1e-4, 1e-1)
        >>>     config["data"]["batch_size"] = tune.choice([16, 32, 64, 128])
        >>>     return config
        >>> # Training function.
        >>> def train_fn(config, train=None, val=None):
        >>>     config = OmegaConf.create(config) # convert dict from ray tune to DictConfig
        >>>     ldm = PLDataModuleFromDatasets(train, val=val, seed=config.seed, no_test_set=True, **config.data)
        >>>     model = Net(**config.model)
        >>>     optimizer = getattr(optim, config.optimizer)(model.parameters(), **config.optim)
        >>>     criterion = nn.CrossEntropyLoss()
        >>>     lm = PLModule(
        >>>         model, optimizer, criterion,
        >>>         hparams=config,
        >>>         metrics={"acc": FromLogits(pl.metrics.classification.Accuracy())}, # Logs train_acc and val_acc
        >>>     )
        >>>     metrics_map = {"accuracy": "val_acc", "validation_loss": "val_loss"}  # map metrics from pl to ray tune
        >>>     trainer = make_trainer_for_ray_tune(metrics_map=metrics_map, **config.trainer)
        >>>     trainer.fit(lm, datamodule=ldm)
        >>> # Run optimization
        >>> if __name__ == "__main__":
        >>>     config, train_dataset, val_dataset = ...
        >>>     best_config = run_tuning(
        >>>         config,
        >>>         "configs/best.tuning.config.yml",
        >>>         train_fn,
        >>>         configure_search_space,
        >>>         train_dataset,
        >>>         val_dataset,
        >>>     )
    """
    config = _extract_wandb_config(config)
    cfg = config_fn(cast(Dict[str, Any], OmegaConf.to_container(config)))
    cfg["trainer"]["gpus"] = math.ceil(cfg["tune"]["gpus_per_trial"])
    trainable = tune.with_parameters(train_fn, train=train, val=val)
    metric, mode = cfg["tune"]["metric"], cfg["tune"]["mode"]

    analysis = tune.run(
        trainable,
        loggers=[
            WandbLogger
        ],  # WandbLogger logs experiment configurations and metrics reported via tune.report() to W&B Dashboard
        resources_per_trial={
            "cpu": cfg["tune"]["cpus_per_trial"],
            "gpu": cfg["tune"]["gpus_per_trial"],
        },
        config=cfg,
        max_failures=10,
        num_samples=cfg["tune"]["num_trials"],
        search_alg=OptunaSearch(metric=metric, mode=mode),
        metric=metric,
        mode=mode,
        # scheduler=tune.schedulers.ASHAScheduler(metric=metric, mode=mode, reduction_factor=2),
        name=f"{cfg['trainer']['experiment_name']}-tuning",
    )
    best_config = analysis.get_best_config(metric, mode)
    best_result = analysis.get_best_trial(metric=metric, mode=mode).last_result
    logger.info(f"Best hyperparameters found were: {best_config}")
    logger.info(f"Best score: {best_result[metric]}")

    best_config["tune"]["result"] = best_result

    yaml_dump(best_config, output_config_file)

    return best_config
