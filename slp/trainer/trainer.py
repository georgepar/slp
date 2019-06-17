import torch
import torch.nn as nn

from ignite.handlers import EarlyStopping
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, State
from ignite.metrics import RunningAverage, Loss

from torch.optim.optimizer import Optimizer
from torch.nn.loss import _Loss
from torch.utils.data import DataLoader

from typing import cast, List, Optional, Tuple, TypeVar
from slp.util import types

from slp.trainer.handlers import CheckpointHandler, EvaluationHandler
from slp.util import from_checkpoint, to_device
from slp.util import log
from slp.util import system

LOGGER = log.getLogger('default')

TrainerType = TypeVar('TrainerType', bound='Trainer')


class Trainer(object):
    def __init__(self: TrainerType,
                 model: nn.Module,
                 optimizer: Optimizer,
                 checkpoint_dir: str = '../../checkpoints',
                 experiment_name: str = 'experiment',
                 model_checkpoint: Optional[str] = None,
                 optimizer_checkpoint: Optional[str] = None,
                 metrics: types.GenericDict = None,
                 patience: int = 10,
                 validate_every: int = 1,
                 accumulation_steps: int = 1,
                 loss_fn: _Loss = nn.CrossEntropyLoss(),
                 non_blocking: bool = True,
                 dtype: torch.dtype = torch.float,
                 device: str = 'cpu') -> None:
        self.dtype = dtype
        self.non_blocking = non_blocking
        self.device = device
        self.loss_fn = loss_fn
        self.validate_every = validate_every
        self.patience = patience
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = checkpoint_dir

        model_checkpoint = self._check_checkpoint(model_checkpoint)
        optimizer_checkpoint = self._check_checkpoint(optimizer_checkpoint)

        if metrics is None:
            metrics = {}
        if 'loss' not in metrics:
            metrics['loss'] = Loss(self.loss_fn)
        self.model = cast(nn.Module, from_checkpoint(
                model_checkpoint, model, map_location=torch.device('cpu')))
        self.model = self.model.type(dtype).to(device)
        self.optimizer = from_checkpoint(optimizer_checkpoint, optimizer)

        self.trainer = Engine(self.train_step)
        self.train_evaluator = Engine(self.eval_step)
        self.valid_evaluator = Engine(self.eval_step)
        for name, metric in metrics.items():
            metric.attach(self.train_evaluator, name)
            metric.attach(self.valid_evaluator, name)

        self.pbar = ProgressBar()
        self.val_pbar = ProgressBar(desc='Validation')

        self.checkpoint = CheckpointHandler(
            checkpoint_dir, experiment_name, score_name='validation_loss',
            score_function=self._score_fn, n_saved=2,
            require_empty=False, save_as_state_dict=True)

        self.early_stop = EarlyStopping(
            patience, self._score_fn, self.trainer)

        self.val_handler = EvaluationHandler(pbar=self.pbar,
                                             validate_every=1,
                                             early_stopping=self.early_stop)
        self.attach()

    def _check_checkpoint(self: TrainerType,
                          ckpt: Optional[str]) -> Optional[str]:
        if system.is_url(ckpt):
            ckpt = system.download_url(cast(str, ckpt), self.checkpoint_dir)
        return ckpt

    @staticmethod
    def _score_fn(engine: Engine) -> float:
        """Returns the scoring metric for checkpointing and early stopping

        Args:
            engine (ignite.engine.Engine): The engine that calculates
            the val loss

        Returns:
            (float): The validation loss
        """
        negloss: float = -engine.state.metrics['loss']
        return negloss

    def parse_batch(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        targets = to_device(batch[1],
                            device=self.device,
                            non_blocking=self.non_blocking)
        return inputs, targets

    def get_predictions_and_targets(
            self: TrainerType,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs, targets = self.parse_batch(batch)
        y_pred = self.model(inputs)
        return y_pred, targets

    def train_step(self: TrainerType,
                   engine: Engine,
                   batch: List[torch.Tensor]) -> float:
        self.model.train()
        y_pred, targets = self.get_predictions_and_targets(batch)
        loss = self.loss_fn(y_pred, targets)
        loss = loss / self.accumulation_steps
        loss.backward()
        if (self.trainer.state.iteration + 1) % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        loss_value: float = loss.item()
        return loss_value

    def eval_step(
            self: TrainerType,
            engine: Engine,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        self.model.eval()
        with torch.no_grad():
            y_pred, targets = self.get_predictions_and_targets(batch)
            return y_pred, targets

    def predict(self: TrainerType, dataloader: DataLoader) -> State:
        return self.valid_evaluator.run(dataloader)

    def fit(self: TrainerType,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 50) -> State:
        self.val_handler.attach(self.trainer,
                                self.train_evaluator,
                                train_loader,
                                validation=False)
        self.val_handler.attach(self.trainer,
                                self.valid_evaluator,
                                val_loader,
                                validation=True)
        self.model.zero_grad()
        self.trainer.run(train_loader, max_epochs=epochs)

    def attach(self: TrainerType) -> TrainerType:
        ra = RunningAverage(output_transform=lambda x: x)
        ra.attach(self.trainer, "Train Loss")
        self.pbar.attach(self.trainer, ['Train Loss'])
        self.val_pbar.attach(self.train_evaluator)
        self.val_pbar.attach(self.valid_evaluator)
        self.valid_evaluator.add_event_handler(Events.COMPLETED,
                                               self.early_stop)
        ckpt = {
            'model': self.model,
            'optimizer': self.optimizer
        }
        self.valid_evaluator.add_event_handler(Events.COMPLETED,
                                               self.checkpoint,
                                               ckpt)

        def graceful_exit(engine, e):
            if isinstance(e, KeyboardInterrupt):
                engine.terminate()
                LOGGER.warn("CTRL-C caught. Exiting gracefully...")
            else:
                raise(e)

        self.trainer.add_event_handler(Events.EXCEPTION_RAISED, graceful_exit)
        self.train_evaluator.add_event_handler(Events.EXCEPTION_RAISED,
                                               graceful_exit)
        self.valid_evaluator.add_event_handler(Events.EXCEPTION_RAISED,
                                               graceful_exit)
        return self


class AutoencoderTrainer(Trainer):
    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        return inputs, inputs


class SequentialTrainer(Trainer):
    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        targets = to_device(batch[1],
                            device=self.device,
                            non_blocking=self.non_blocking)
        lengths = to_device(batch[2],
                            device=self.device,
                            non_blocking=self.non_blocking)
        return inputs, targets, lengths

    def get_predictions_and_targets(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs, targets, lengths = self.parse_batch(batch)
        y_pred = self.model(inputs, lengths)
        return y_pred, targets


class Seq2seqTrainer(SequentialTrainer):
    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        lengths = to_device(batch[1],
                            device=self.device,
                            non_blocking=self.non_blocking)
        return inputs, inputs, lengths


class TransformerTrainer(Trainer):
    def parse_batch(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        targets = to_device(batch[1],
                            device=self.device,
                            non_blocking=self.non_blocking)
        mask_inputs = to_device(batch[2],
                                device=self.device,
                                non_blocking=self.non_blocking)
        mask_targets = to_device(batch[3],
                                 device=self.device,
                                 non_blocking=self.non_blocking)
        return inputs, targets, mask_inputs, mask_targets

    def get_predictions_and_targets(
            self,
            batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs, targets, mask_inputs, mask_targets = self.parse_batch(batch)
        y_pred = self.model(inputs,
                            targets,
                            source_mask=mask_inputs,
                            target_mask=mask_targets)
        targets = targets.view(-1)
        y_pred = y_pred.view(targets.size(0), -1)
        # TODO: BEAMSEARCH!!
        return y_pred, targets
