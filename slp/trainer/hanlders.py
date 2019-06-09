import shutil
from ignite.handlers import ModelCheckpoint

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
            dst = '{}_{}.best.pth'.format(fname_prefix, name)
            shutil.copy(src, dst)