import os
from typing import Any, Callable, Dict, Union, cast

from loguru import logger
from omegaconf import DictConfig, ListConfig
from ray import tune
from ray.tune.integration.wandb import WandbLogger
from ray.tune.suggest.optuna import OptunaSearch
from slp.config.omegaconf import OmegaConfExtended as OmegaConf
from slp.util.system import date_fname, has_internet_connection, safe_mkdirs, yaml_dump


def extract_wandb_config(config: DictConfig) -> DictConfig:
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
    train: Any,
    val: Any,
):
    config = extract_wandb_config(config)
    cfg = config_fn(cast(Dict[str, Any], OmegaConf.to_container(config)))
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
        num_samples=cfg["tune"]["num_trials"],
        search_alg=OptunaSearch(metric=metric, mode=mode),
        scheduler=tune.schedulers.ASHAScheduler(metric=metric, mode=mode),
        name=f"{cfg['trainer']['experiment_name']}-tuning",
    )

    best_config = analysis.get_best_config(metric, mode)
    best_result = analysis.get_best_trial(metric=metric, mode=mode).last_result
    logger.info("Best hyperparameters found were: ", best_config)
    logger.info("Best score: ", best_result)

    best_config["tune"]["result"] = best_result

    yaml_dump(best_config, output_config_file)

    return best_config
