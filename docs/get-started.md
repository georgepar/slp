# Getting started

For a quick hands-on we are going to go through creating a MNIST classifier step by step (`examples/mnist.py`).

First we can see the model definition of a simple CNN classifier:

```python
class Net(nn.Module):
    def __init__(self, intermediate_hidden=50):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, intermediate_hidden)
        self.fc2 = nn.Linear(intermediate_hidden, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
```

The `get_data` function downloads MNIST and performs the preprocessing:

```python
def get_data():
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train = MNIST(download=True, root=".", transform=data_transform, train=True)

    val = MNIST(download=False, root=".", transform=data_transform, train=False)
    return train, val
```

Finally the `get_parser` function creates a CLI parser for the model arguments. Note the destination variable `model.intermediate_hidden` (important for later):

```python
def get_parser():
    parser = ArgumentParser("MNIST classification example")
    parser.add_argument(
        "--hidden",
        dest="model.intermediate_hidden",
        type=int,
        default=12,
        help="Intermediate hidden layers for linear module",
    )
    return parser
```

## Setup

You would create these functions whether you use slp or not. The interesting part is in the main function.

Here we first need to perform some setup for logging, configuration parsing and seeding. Let's go through this step-by-step:

First we create the CLI args parser. `make_cli_parser` takes the parser we defined in get_parser() and extends it with generic arguments for the data, optimizers, learning rate scheduling, training and experiment tracking.

Run `python mnist.py --help` for a full list of arguments, or go down in the Appendix. 
The arguments have a detailed description and most of them default to `None` so they will not be used.

```python
parser = get_parser()
parser = make_cli_parser(parser, PLDataModuleFromDatasets)
```

Next we parse the configuration. Here we need to provide the parser and (optionally) a YAML configuration file `python mnist.py --config my-config.yaml`.

The configuration file should have the following format:

```yaml
model:
  intermediate_hidden: 100
optimizer: Adam
optim:
  lr: 1e-3
lr_scheduler: true  # ReduceLROnPlateau
lr_schedule:
  factor: 2
data:
  batch_size: 128
  batch_size_eval: 256
```

Note that this format, closely follows the `dest` values we configure in the command line args (e.g. `OPTIM.LR`, `LR_SCHEDULE.FACTOR`), namely the dots. form a hierarchy.

This way we can use `parse_config` to merge the values in the configuration file and the CLI args.

The precedence is as follows:

```
default CLI args < config file values < user provided CLI args
```

So if we call the script with `--lr 1e-4` this value will overwrite the value in the configuration file.
If a value is not specified in the configuration file, the default value we specified in argparse will be set.
If a value is not specified in any of these places, sane defaults will be used.

```python
config = parse_config(parser, parser.parse_args().config)

if config.trainer.experiment_name == "experiment":
    config.trainer.experiment_name = "mnist-classification"
```

Next, we configure logging. This call configures `loguru` to intercept all logs and print the both to stdout and a log file.
The log file name will depend on the experiment name we provided and `datetime.now()`, to avoid overwriting previous runs (e.g. `mnist-classification.20210302-134714.log`).

```python
configure_logging(f"logs/{config.trainer.experiment_name}")
```

Finally, we make the run deterministic (`--seed`)

```python
if config.seed is not None:
    logger.info("Seeding everything with seed={seed}")
    pl.utilities.seed.seed_everything(seed=config.seed)
```

## Data Module

Here we download the train and test datasets and define the `LightningDataModule` that will be used in this experiment.

The `LightningDataModule` is the preferred way to consume datasets in pytorch lightning, and `PLDataModuleFromDatasets` abstracts the boilerplate of constructing and configuring the `DataLoaders`, splitting data etc.

**Note**: `PLDataModuleFromDatasets` expects three `torch.utils.data.Datasets` as input, train, val and test. Val and test are optional. If any of the validation or tests sets are not provided, `PLDataModuleFromDatasets` will create a split using 20% of the train set by default (see `--val-percent`, `--test-percent`).

```python
train, test = get_data()

ldm = PLDataModuleFromDatasets(train, test=test, seed=config.seed, **config.data)  # Note we pass **config.data, because config is in a hierarchy.
```

## Defining the model

Next we define the model, optimizer, criterion and learning rate scheduler (pretty standard).

```python
model = Net(**config.model)

optimizer = getattr(optim, config.optimizer)(model.parameters(), **config.optim)
criterion = nn.CrossEntropyLoss()

lr_scheduler = None
if config.lr_scheduler:
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **config.lr_schedule
    )
```

And the `LightningModule` that will be used for training. This module takes care of defining the train and validation steps, computing and logging metrics etc.

Note that pytorch lightning by default expects softmaxed outputs in the predefined metrics, so we wrap the metric with `FromLogits` to use crossentropy loss.

```python
lm = PLModule(
    model,
    optimizer,
    criterion,
    lr_scheduler=lr_scheduler,
    metrics={"acc": FromLogits(pl.metrics.classification.Accuracy())},
    hparams=config,  # We pass this so that configuration will be logged in wandb
)
```


## Training and Debugging

Finally, we have the option to run a full training and testing phase of the model, or run a quick debug execution.

If we need to debug, we can pass `--debug` flag, and the trainer will run a full training, validation run on 5 batches.
It will also try to overfit the model on 5 batches to verify that gradients flow.

```python
# Run debugging session or fit & test the model ############
if config.debug:
    logger.info("Running in debug mode: Fast run on 5 batches")
    trainer = make_trainer(fast_dev_run=5)
    trainer.fit(lm, datamodule=ldm)

    logger.info("Running in debug mode: Overfitting 5 batches")
    trainer = make_trainer(overfit_batches=5)
    trainer.fit(lm, datamodule=ldm)
```

If we run in normal mode, we fit on train / val sets and evaluate the best model on the test set. The best model is selected as the model with the smallest validation loss.

Training will run with early stopping, the best 3 checkpoints will be saved, all the jazz.

Note `watch_model` tells wandb to track weight norms and gradients for further inspection.

```python
else:
    trainer = make_trainer(**config.trainer)
    watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)

    trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())

    logger.info("Run finished. Uploading files to wandb...")
```


Sure, most of this goodness comes from the awesome team in pytorch lightning. But what we do here is we abstract the boilerplate and the large learning curve, without making sacrifices in the features.

For example, you can play spot the differences between `examples/mnist.py`, which performs digit classification and `examples/smt_bert.py` which finetunes BERT for sentiment classification on SST-2.

The classes we use change, but the way they are called, the structure and features remains the same.

```python
# smt_bert.py
...
if __name__ == "__main__":
    parser = get_parser()
    parser = make_cli_parser(parser, PLDataModuleFromCorpus)

    args = parser.parse_args()
    config_file = args.config

    config = parse_config(parser, config_file)
    # Set these by default.
    config.hugging_face_model = config.data.tokenizer
    config.data.add_special_tokens = True
    config.data.lower = "uncased" in config.hugging_face_model

    if config.trainer.experiment_name == "experiment":
        config.trainer.experiment_name = "finetune-bert-smt"

    configure_logging(f"logs/{config.trainer.experiment_name}")

    if config.seed is not None:
        logger.info("Seeding everything with seed={seed}")
        pl.utilities.seed.seed_everything(seed=config.seed)

    (
        raw_train,
        labels_train,
        raw_dev,
        labels_dev,
        raw_test,
        labels_test,
        num_labels,
    ) = get_data(config)

    ldm = PLDataModuleFromCorpus(
        raw_train,
        labels_train,
        val=raw_dev,
        val_labels=labels_dev,
        test=raw_test,
        test_labels=labels_test,
        collate_fn=collate_fn,
        **config.data,
    )

    model = BertForSequenceClassification.from_pretrained(
        config.hugging_face_model, num_labels=num_labels
    )

    logger.info(model)

    # Leave this hardcoded for now.
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    lm = BertPLModule(
        model,
        optimizer,
        criterion,
        metrics={"acc": FromLogits(pl.metrics.classification.Accuracy())},
    )

    trainer = make_trainer(**config.trainer)
    watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)

    trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())

```

## Appendix. Command Line arguments

```
usage: mnist.py [-h] [--hidden MODEL.INTERMEDIATE_HIDDEN]
                [--optimizer {Adam,AdamW,SGD,Adadelta,Adagrad,Adamax,ASGD,RMSprop}] [--lr OPTIM.LR]
                [--weight-decay OPTIM.WEIGHT_DECAY] [--lr-scheduler]
                [--lr-factor LR_SCHEDULE.FACTOR] [--lr-patience LR_SCHEDULE.PATIENCE]
                [--lr-cooldown LR_SCHEDULE.COOLDOWN] [--min-lr LR_SCHEDULE.MIN_LR] [--seed SEED]
                [--config CONFIG] [--experiment-name TRAINER.EXPERIMENT_NAME]
                [--run-id TRAINER.RUN_ID] [--experiment-group TRAINER.EXPERIMENT_GROUP]
                [--experiments-folder TRAINER.EXPERIMENTS_FOLDER] [--save-top-k TRAINER.SAVE_TOP_K]
                [--patience TRAINER.PATIENCE] [--wandb-project TRAINER.WANDB_PROJECT]
                [--tags [TRAINER.TAGS [TRAINER.TAGS ...]]] [--stochastic_weight_avg]
                [--gpus TRAINER.GPUS] [--val-interval TRAINER.CHECK_VAL_EVERY_N_EPOCH]
                [--clip-grad-norm TRAINER.GRADIENT_CLIP_VAL] [--epochs TRAINER.MAX_EPOCHS]
                [--steps TRAINER.MAX_STEPS] [--tbtt_steps TRAINER.TRUNCATED_BPTT_STEPS] [--debug]
                [--val-percent DATA.VAL_PERCENT] [--test-percent DATA.TEST_PERCENT]
                [--bsz DATA.BATCH_SIZE] [--bsz-eval DATA.BATCH_SIZE_EVAL]
                [--num-workers DATA.NUM_WORKERS] [--pin-memory] [--drop-last] [--shuffle-eval]

optional arguments:
  -h, --help            show this help message and exit
  --hidden MODEL.INTERMEDIATE_HIDDEN
                        Intermediate hidden layers for linear module
  --optimizer {Adam,AdamW,SGD,Adadelta,Adagrad,Adamax,ASGD,RMSprop}
                        Which optimizer to use
  --lr OPTIM.LR         Learning rate
  --weight-decay OPTIM.WEIGHT_DECAY
                        Learning rate
  --lr-scheduler        Use learning rate scheduling. Currently only ReduceLROnPlateau is supported
                        out of the box
  --lr-factor LR_SCHEDULE.FACTOR
                        Multiplicative factor by which LR is reduced. Used if --lr-scheduler is
                        provided.
  --lr-patience LR_SCHEDULE.PATIENCE
                        Number of epochs with no improvement after which learning rate will be
                        reduced. Used if --lr-scheduler is provided.
  --lr-cooldown LR_SCHEDULE.COOLDOWN
                        Number of epochs to wait before resuming normal operation after lr has been
                        reduced. Used if --lr-scheduler is provided.
  --min-lr LR_SCHEDULE.MIN_LR
                        Minimum lr for LR scheduling. Used if --lr-scheduler is provided.
  --seed SEED           Seed for reproducibility
  --config CONFIG       Path to YAML configuration file
  --experiment-name TRAINER.EXPERIMENT_NAME
                        Name of the running experiment
  --run-id TRAINER.RUN_ID
                        Unique identifier for the current run. If not provided it is inferred from
                        datetime.now()
  --experiment-group TRAINER.EXPERIMENT_GROUP
                        Group of current experiment. Useful when evaluating for different seeds /
                        cross-validation etc.
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
  --debug               If true, we run a full run on a small subset of the input data and overfit
                        10 training batches
  --val-percent DATA.VAL_PERCENT
                        Percent of validation data to be randomly split from the training set, if
                        no validation set is provided
  --test-percent DATA.TEST_PERCENT
                        Percent of test data to be randomly split from the training set, if no test
                        set is provided
  --bsz DATA.BATCH_SIZE
                        Training batch size
  --bsz-eval DATA.BATCH_SIZE_EVAL
                        Evaluation batch size
  --num-workers DATA.NUM_WORKERS
                        Number of workers to be used in the DataLoader
  --pin-memory          Pin data to GPU memory for faster data loading
  --drop-last           Drop last incomplete batch
  --shuffle-eval        Shuffle val & test sets
```