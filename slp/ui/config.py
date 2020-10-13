import os

from slp.config import BASE_PATH
from slp.ui import cli
from slp.util.system import yaml_load


SANE_DEFAULTS = {
    'device': 'cpu',
    'train': True,
    'test': True,
    'data_dir': os.path.join(BASE_PATH, 'data'),
    'cache_dir': os.path.join(BASE_PATH, 'cache'),
    'logging_dir': os.path.join(BASE_PATH, 'logs'),
    'optimizer': {
        'name': 'Adam',
        'learning_rate': 1e-3,
    },
    'dataloaders': {
        'num_workers': 1,
        'pin_memory': True,
    },
    'trainer': {
        'accumulation_steps': 1,
        'patience': 5,
        'max_epochs': 100,
        'validate_every': 1,
        'parallel': False,
        'non_blocking': True,
        'retain_graph': False,
        'checkpoint_dir': os.path.join(BASE_PATH, 'checkpoints'),
        'model_checkpoint': None,
        'optimizer_checkpoint': None,
    }
}


def _merge(*dicts):
    if len(dicts) == 1:
        return dicts[0]
    merged = dicts[0]
    for d in reversed(dicts[1:]):
        for k, v in d.items():
            merged[k] = v
            if isinstance(v, dict):
                d[k] = _merge(*[subd[k] for subd in dicts if k in subd])
            else:
                continue
    return merged


def load_config(parser=None):
    '''Load yaml configuration and overwrite with CLI args if provided
    Configuration file format:
        experiment:
            name: "imdb-glove-rnn-256-bi'
            description: Long experiment description
        embeddings:
            path: "../data/glove.840B.300d.txt"
            dim: 300
        dataloaders:
            batch_size: 32
            pin_memory: True
        models:
            rnn:
                hidden_size: 256
                layers: 1
                bidirectional: True
                attention: True
            classifier:
                in_features: 512
                num_classes: 3
        optimizer:
            name: Adam
            learning_rate: 1e-3
        loss:
            name: CrossEntropyLoss
        trainer:
            patience: 5
            retain_graph: True
    '''
    if parser is None:
        cli_args = cli.default_cli()
    else:
        cli_args = cli.get_cli(parser)
    config_file = cli_args['config']
    cfg = yaml_load(config_file)
    return _merge(cli_args, cfg, SANE_DEFAULTS)


if __name__ == '__main__':
    cfg = load_config('../../tests/test.yaml')
    import pprint
    pprint.pprint(cfg)

