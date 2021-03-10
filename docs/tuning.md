# Hyperparameter tuning

We provide easy distributed hyperparameter tuning using Ray Tune. Let's walk through the
`examples/mnist_tune.py` script for a concrete example.

This script has been modified from `examples/mnist.py` that was presented in
[Getting started](getting-started.md), to perform a distributed hyperparameter parameter tuning
run using Ray Tune and the SLP utilities.

First we refactor the model creation and training into a function, so that each worker is able to
instantiate and train a model.

```python
from slp.plbind.trainer import make_trainer_for_ray_tune
...
def train_mnist(config, train=None, val=None):
    # Convert dictionary to omegaconf dictconfig object
    config = OmegaConf.create(config)

    # Create data module
    ldm = PLDataModuleFromDatasets(
        train, val=val, seed=config.seed, no_test_set=True, **config.data
    )

    # Create model, optimizer, criterion, scheduler
    model = Net(**config.model)

    optimizer = getattr(optim, config.optimizer)(model.parameters(), **config.optim)
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = None

    if config.lr_scheduler:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config.lr_schedule
        )

    # Wrap in PLModule, & configure metrics
    lm = PLModule(
        model,
        optimizer,
        criterion,
        lr_scheduler=lr_scheduler,
        metrics={
            "acc": FromLogits(pl.metrics.classification.Accuracy())
        },  # Will log train_acc and val_acc
        hparams=config,
    )

    # Map Lightning metrics to ray tune metris.
    metrics_map = {"accuracy": "val_acc", "validation_loss": "val_loss"}
    assert (
        config["tune"]["metric"] in metrics_map.keys()
    ), "Metrics mapping should contain the metric you are trying to optimize"
    # Train model
    trainer = make_trainer_for_ray_tune(metrics_map=metrics_map, **config.trainer)

    trainer.fit(lm, datamodule=ldm)
```

This function is pretty similar to the `examples/mnist.py` code, but there are some important notes to keep in mind here.

* The function accepts a config dictionary as a positional argument and the train and validation
  datasets as keyword arguments. This is important, because `run_tuning` expects the input training
  function to have this stub.
* We convert the input dict to an omegaconf.DictConfig object for convenience. Unfortunately ray
  tune is not able to pass around OmegaConf configuration objects so we need to convert back and
  forth.
* Take note of the `metrics_map`. This mapping renames the metrics aggregated by our pytorch
  lightning module so that they are logged for ray tune. One of the keys of this dict is going to
  be chosen as the metric to optimize
* We use `the make_trainer_for_ray_tune` To create a trainer that is configured specifically for a
  tuning run


Next we define the `configure_search_space` function, that overrides entries in the configuration
file with ranges, from which ray tune will sample values for the hyperparameters.

```python
def configure_search_space(config):
    config["model"] = {
        "intermediate_hidden": tune.choice([16, 32, 64, 100, 128, 256, 300, 512])
    }
    config["optimizer"] = tune.choice(["SGD", "Adam", "AdamW"])
    config["optim"]["lr"] = tune.loguniform(1e-4, 1e-1)
    config["optim"]["weight_decay"] = tune.loguniform(1e-4, 1e-1)
    config["data"]["batch_size"] = tune.choice([16, 32, 64, 128])

    return config
```

As you can see, we are going to tune the learning rate, weight decay, optimizer,
batch size and the hidden size of our model.

**Note**: I considered abstracting this into a configuration
file, but I don't have any use case for this kind of abstraction and the simplicity and flexibility
we get from keeping this in code is more important.

Finally we parse the config file and the CLI arguments and spawn the hyperparameter tuning in the
main function:


```python
from slp.util.tuning import run_tuning
...
if __name__ == "__main__":
    config = ...
    train, val, _ = get_data()
    best_config = run_tuning(
        config,
        "configs/best.mnist.tune.yml",
        train_mnist,
        configure_search_space,
        train,
        val,
    )
```

The `run_tuning` function accepts
* The parsed configuration as an `omegaconf.DictConfig` object
* A path to save the best trial configuration as a yaml file
* The `train_mnist` function
* The `configure_search_space` function
* The `train` dataset
* The `validation` dataset

Note that we create the train and validation splits by hand, so that each trial runs on the same
validation set.

The script can be called with the following arguments:

```bash
python examples/minst_tune.py --num-trials 1000 --cpus_per_trial 1 --gpus_per_trial 0.12 --tune-metric accuracy --tune-mode max --epochs 20
```

This will spawn 1000 trials over as many gpus as we have available (in our server or in our cluster).
The `--gpus_per_trial` argument can be a floating point number. In this case we pack 7-8
experiments per GPU (RTX 2080Ti).

Note the `--tune-metric` argument corresponds to one of the keys in the `metrics_map` dictionary.
Here we run the tuning to optimize for validation accuracy.

After the run finishes we can use the `configs/best.mnist.tune.yml` configuration file to train and
evaluate the best model on the test set

```bash
python examples/mnist.py --config configs/best.mnist.tune.yml
```

**Note**: `mnist_tune.py` can accept any configuration file and command line argument that `mnist.py` can accept. Run `python minst_tune.py --help` for more information.

::: slp.util.tuning
