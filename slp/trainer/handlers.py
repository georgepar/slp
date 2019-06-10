import shutil

from ignite.engine import Events
from ignite.handlers import ModelCheckpoint


from slp.util import system as sysutil


class CheckpointHandler(ModelCheckpoint):
    """Augment ignite ModelCheckpoint Handler with copying the best file to a
    {filename_prefix}_{experiment_name}.best.pth.
    This helps for automatic testing etc.
    Args:
        engine (ignite.engine.Engine): The trainer engine
        to_save (dict): The objects to save
    """
    def __call__(self, engine, to_save):
        super(CheckpointHandler, self).__call__(engine, to_save)
        # Select model with best loss
        _, paths = self._saved[-1]
        for src in paths:
            splitted = src.split('_')
            fname_prefix = splitted[0]
            name = splitted[1]
            dst = f'{fname_prefix}_{name}.best.pth'
            shutil.copy(src, dst)


class EvaluationHandler(object):
    def __init__(self, pbar=None, validate_every=1, early_stopping=None):
        self.validate_every = validate_every
        self.print_fn = pbar.log_message if pbar is not None else print
        self.early_stopping = early_stopping

    def __call__(self, engine, evaluator, dataloader, validation=True):
        if engine.state.epoch % self.validate_every != 0:
            return
        evaluator.run(dataloader)
        sysutil.print_separator(n=35, print_fn=self.print_fn)
        metrics = evaluator.state.metrics
        phase = 'Validation' if validation else 'Training'
        self.print_fn('Epoch {} {} results'
                      .format(engine.state.epoch, phase))
        sysutil.print_separator(symbol='-', n=35, print_fn=self.print_fn)
        for name, value in metrics.items():
            self.print_fn('{:<15} {:<15}'.format(name, value))

        if validation and self.early_stopping:
            loss = self.early_stopping.best_score
            patience = self.early_stopping.patience
            cntr = self.early_stopping.counter
            self.print_fn('{:<15} {:<15}'.format('best loss', -loss))
            self.print_fn('{:<15} {:<15}'.format('patience left',
                                                 patience - cntr))
            sysutil.print_separator(n=35, print_fn=self.print_fn)

    def attach(self, trainer, evaluator, dataloader, validation=True):
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            self, evaluator, dataloader,
            validation=validation)
