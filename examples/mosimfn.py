import h5py
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.metrics import Loss, Accuracy
from pprint import pprint
from torch.utils.data import DataLoader

from slp.data.mosi import MOSIMFN
from slp.data.collators import MOSICollator
from slp.data.transforms import InstanceNorm, ToTensor
from slp.modules.multimodal import AudioGloveClassifier, GloveClassifier, AudioClassifier
from slp.trainer import MOSITrainer
from slp.ui.config import load_config
from slp.util import log


def load_saved_data(path):
    h5f = h5py.File(os.path.join(path, 'X_train.h5'), 'r')
    X_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(os.path.join(path, 'y_train.h5'), 'r')
    y_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(os.path.join(path, 'X_valid.h5'), 'r')
    X_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(os.path.join(path, 'y_valid.h5'), 'r')
    y_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(os.path.join(path, 'X_test.h5'), 'r')
    X_test = h5f['data'][:]
    h5f.close()
    h5f = h5py.File(os.path.join(path, 'y_test.h5'), 'r')
    y_test = h5f['data'][:]
    h5f.close()
    return X_train, y_train, X_valid, y_valid, X_test, y_test


class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()

    def forward(self, out, tgt):
        tgt = tgt.view(-1, 1).float()
        return F.binary_cross_entropy_with_logits(out, tgt)


def get_parser():
    parser = argparse.ArgumentParser(description='CLI parser for experiment')

    parser.add_argument(
        '--binary', dest='binary',
        default=False,
        action='store_true',
        help='Use binary task')

    parser.add_argument(
        '--remove-pauses', dest='remove_pauses',
        default=False,
        action='store_true',
        help='Remove speaker pauses')

    parser.add_argument(
        '--modalities', dest='modalities',
        default=None, nargs='+',
        help='Modalities to use')

    parser.add_argument(
        '--audio-dim', dest='audio.dim',
        default=None, type=int,
        help='Audio features dimension')

    parser.add_argument(
        '--audio-instance-norm', dest='audio.instance_norm',
        action="store_true",
        help='Audio instance normalization')

    parser.add_argument(
        '--audio-batchnorm', dest='audio.batchnorm',
        action="store_true",
        help='Audio batch normalization')

    parser.add_argument(
        '--audio-layers', dest='audio.layers',
        default=None, type=int,
        help='Num layers for audio encoder')

    parser.add_argument(
        '--text-layers', dest='text.layers',
        default=None, type=int,
        help='Num layers for text encoder')

    parser.add_argument(
        '--audio-hidden-size', dest='audio.hidden_size',
        default=None, type=int,
        help='Hidden size for RNNs')

    parser.add_argument(
        '--text-hidden-size', dest='text.hidden_size',
        default=None, type=int,
        help='Hidden size for RNNs')

    parser.add_argument(
        '--dropout', dest='common.dropout',
        default=None, type=float,
        help='Dropout probabiity')

    parser.add_argument(
        '--proj-size', dest='fuse.projection_size',
        default=None, type=int,
        help='Modality projection size')

    parser.add_argument(
        '--bidirectional', dest='common.bidirectional',
        action="store_true",
        help='Use BiRNNs')

    parser.add_argument(
        '--rnn-type', dest='common.rnn_type',
        default=None, type=str,
        help='lstm or gru')

    parser.add_argument(
        '--text-attention', dest='text.attention',
        action="store_true",
        help='Use RNN with attention')

    parser.add_argument(
        '--audio-attention', dest='audio.attention',
        action="store_true",
        help='Use RNN with attention')

    parser.add_argument(
        '--fuse', dest='fuse.method',
        default=None, type=str,
        help='cat or add')

    parser.add_argument(
        '--modality-weights', dest='fuse.modality_weights',
        action="store_true",
        help='Use modality weights during fusion')
    return parser


C = load_config(parser=get_parser())
C['modalities'] = set(C['modalities'])

collate_fn = MOSICollator(
    device='cpu',
    binary=C['binary'],
    modalities=C['modalities']
)


if __name__ == "__main__":
    log.info('Running with configuration')
    pprint(C)
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_saved_data(C['data_dir'])
    to_tensor = ToTensor(device="cpu")
    to_tensor_float = ToTensor(device="cpu", dtype=torch.float)
    instance_norm = InstanceNorm()

    def create_dataloader(dat):
        X, y = dat
        d = (
            MOSIMFN(X, y, binary=C['binary'], modalities=C['modalities'])
            .map(to_tensor_float, 'text', lazy=True)
        )
        if C["audio"]["instance_norm"]:
            d = d.map(instance_norm, "audio", lazy=True)
            d = d.map(instance_norm, 'visual', lazy=True)
        d = d.map(to_tensor_float, 'audio', lazy=True)
        d = d.map(to_tensor_float, 'visual', lazy=True)
        d.apply_transforms()
        return DataLoader(
            d,
            batch_size=C['dataloaders']['batch_size'],
            num_workers=C['dataloaders']['num_workers'],
            pin_memory=C['dataloaders']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
        )

    train_loader, dev_loader, test_loader = map(
        create_dataloader, [(X_train, y_train), (X_dev, y_dev), (X_test, y_test)]
    )
    x = next(iter(train_loader))

    num_classes = 1 if C["binary"] else 5
    model = AudioClassifier(
        C["audio"]["dim"],
        audio_hidden_size=C["audio"]["hidden_size"],
        text_hidden_size=C["text"]["hidden_size"],
        embeddings_dim=C["embeddings"]["dim"],
        batch_first=True,
        text_layers=C["text"]["layers"],
        audio_layers=C["audio"]["layers"],
        audio_batchnorm=C["audio"]["batchnorm"],
        projection_size=C["fuse"]["projection_size"],
        embeddings_dropout=C["embeddings"]["dropout"],
        dropout=C["common"]["dropout"],
        finetune_embeddings=C["embeddings"]["finetune"],
        bidirectional=C["common"]["bidirectional"],
        rnn_type=C["common"]["rnn_type"],
        packed_sequence=True,
        audio_attention=C["audio"]["attention"],
        text_attention=C["text"]["attention"],
        device=C["device"],
        fuse=C["fuse"]["method"],
        modality_weights=C["fuse"]["modality_weights"],
        return_hidden=False, num_classes=num_classes
    )

    optimizer = getattr(torch.optim, C['optimizer']['name'])(
        [p for p in model.parameters() if p.requires_grad],
        lr=C['optimizer']['learning_rate']
    )
    if C['binary']:
        criterion = BCE()
    else:
        criterion = nn.CrossEntropyLoss()

    def thresholded_output_transform(output):
        y_pred, y = output
        y_pred = torch.sigmoid(y_pred)
        y_pred = torch.round(y_pred)
        return y_pred, y

    metrics = {"accuracy": Accuracy(thresholded_output_transform), "loss": Loss(criterion)}

    if C['overfit_batch'] or C['overfit_batch'] or C['train']:
        trainer = MOSITrainer(
            model,
            optimizer,
            experiment_name=C['experiment']['name'],
            checkpoint_dir=C['trainer']['checkpoint_dir'],
            metrics=metrics,
            non_blocking=C['trainer']['non_blocking'],
            patience=C['trainer']['patience'],
            validate_every=C['trainer']['validate_every'],
            retain_graph=C['trainer']['retain_graph'],
            loss_fn=criterion,
            device=C['device'],
        )

    if C['debug']:
        if C['overfit_batch']:
            trainer.overfit_single_batch(train_loader)
        trainer.fit_debug(train_loader, dev_loader)
        sys.exit(0)

    if C['train']:
        trainer.fit(
            train_loader,
            dev_loader,
            epochs=C['trainer']['max_epochs']
        )

    if C['test']:
        trainer = MOSITrainer(
            model,
            optimizer,
            experiment_name=C['experiment']['name'],
            checkpoint_dir=C['trainer']['checkpoint_dir'],
            metrics=metrics,
            model_checkpoint=C['trainer']['load_model'],
            non_blocking=C['trainer']['non_blocking'],
            patience=C['trainer']['patience'],
            validate_every=C['trainer']['validate_every'],
            retain_graph=C['trainer']['retain_graph'],
            loss_fn=criterion,
            device=C['device'],
        )
        pprint(trainer.predict(test_loader).metrics)
