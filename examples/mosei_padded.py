import pickle
import numpy as np
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.metrics import Loss, Accuracy, Fbeta, MeanAbsoluteError, ConfusionMatrix
from pprint import pprint
from torch.utils.data import DataLoader

from slp.data.mosi import MOSEI
from slp.data.collators import MOSICollator
from slp.data.transforms import InstanceNorm, ToTensor
from slp.modules.multimodal import AudioTextClassifier
from slp.trainer import MOSITrainer
from slp.ui.config import load_config
from slp.util import log
from slp.util import split


def load_saved_data(path, normalize=True):
    with open(path, 'rb') as fd:
        train, dev, test = pickle.load(fd)

    if normalize:
        mu_audio = train['audio'].reshape(-1, train['audio'].shape[-1]).mean(0)
        std_audio = train['audio'].reshape(-1, train['audio'].shape[-1]).std(0)
        train['audio'] = (train['audio'] - mu_audio) / (std_audio + 1e-6)
        dev['audio'] = (dev['audio'] - mu_audio) / (std_audio + 1e-6)
        test['audio'] = (test['audio'] - mu_audio) / (std_audio + 1e-6)

        visual_max = np.max(np.max(np.abs(train['visual']), axis=0),axis=0)
        visual_max[visual_max < 1e-7] = 1
        train['visual'] = train['visual'] / (visual_max + 1e-6)
        dev['visual'] = dev['visual'] / (visual_max + 1e-6)
        test['visual'] = test['visual'] / (visual_max + 1e-6)

    return train, dev, test


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
        '--normalize', dest='normalize',
        default=False,
        action='store_true',
        help='Normalize audio and visual features')

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
        '--audio-instance-norm', dest='audio.instance_norm',
        action="store_true",
        help='Audio instance normalization')

    parser.add_argument(
        '--audio-dim', dest='audio.model.input_size',
        default=None, type=int,
        help='Audio features dimension')

    parser.add_argument(
        '--audio-hidden-size', dest='audio.model.hidden_size',
        default=None, type=int,
        help='Hidden size for RNNs')

    parser.add_argument(
        '--audio-layers', dest='audio.model.layers',
        default=None, type=int,
        help='Num layers for audio encoder')

    parser.add_argument(
        '--audio-dropout', dest='audio.model.dropout',
        default=None, type=float,
        help='Dropout probabiity')

    parser.add_argument(
        '--audio-bidirectional', dest='audio.model.bidirectional',
        action="store_true",
        help='Use BiRNN')

    parser.add_argument(
        '--audio-rnn-type', dest='audio.model.rnn_type',
        action="store_true",
        help='RNN type')

    parser.add_argument(
        '--audio-attention', dest='audio.model.attention',
        action="store_true",
        help='Use RNN with attention')

    parser.add_argument(
        '--audio-return-hidden', dest='audio.model.return_hidden',
        action="store_true",
        help='Return hidden states')

    parser.add_argument(
        '--audio-batchnorm', dest='audio.model.batchnorm',
        action="store_true",
        help='Audio batch normalization')

    parser.add_argument(
        '--text-input-size', dest='text.model.input_size',
        default=None, type=int,
        help='Embedding dim')

    parser.add_argument(
        '--text-hidden-size', dest='text.model.hidden_size',
        default=None, type=int,
        help='Hidden size for RNNs')

    parser.add_argument(
        '--text-layers', dest='text.model.layers',
        default=None, type=int,
        help='Num layers for text encoder')

    parser.add_argument(
        '--text-dropout', dest='text.model.dropout',
        default=None, type=float,
        help='Dropout probabiity')

    parser.add_argument(
        '--bidirectional', dest='text.model.bidirectional',
        action="store_true",
        help='Use BiRNNs')

    parser.add_argument(
        '--rnn-type', dest='text.model.rnn_type',
        default=None, type=str,
        help='lstm or gru')

    parser.add_argument(
        '--text-attention', dest='text.model.attention',
        action="store_true",
        help='Use RNN with attention')

    parser.add_argument(
        '--text-return-hidden', dest='text.model.return_hidden',
        action="store_true",
        help='Return hidden RNN states')

    parser.add_argument(
        '--proj-size', dest='fuse.projection_size',
        default=None, type=int,
        help='Modality projection size')

    parser.add_argument(
        '--prefuse', dest='fuse.prefuse',
        action='store_true',
        help='Use tied prefusion layer')

    parser.add_argument(
        '--use-mask', dest='fuse.use_mask',
        action='store_true',
        help='Modality projection size')

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
    binary=False,
    modalities=C['modalities'],
    max_length=-1
)


if __name__ == "__main__":
    log.info('Running with configuration')
    C['binary'] = False
    pprint(C)

    train, dev, test = load_saved_data(C['data_dir'], normalize=C['normalize'])

    to_tensor = ToTensor(device="cpu")
    to_tensor_float = ToTensor(device="cpu", dtype=torch.float)

    def create_dataloader(data):
        dataset = (MOSEI(data, modalities=C['modalities'], unpad=False, select_label=0)
             .map(to_tensor_float, 'text', lazy=True)
        )
        dataset = dataset.map(to_tensor_float, 'audio', lazy=True)
        dataset = dataset.map(to_tensor_float, 'visual', lazy=True)
        dataset.apply_transforms()
        return DataLoader(
            dataset,
            batch_size=C['dataloaders']['batch_size'],
            num_workers=C['dataloaders']['num_workers'],
            pin_memory=C['dataloaders']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
        )
    train_loader, dev_loader, test_loader = map(create_dataloader, [train, dev, test])

    if C['binary']:
        criterion = BCE()
        def thresholded_output_transform(output):
            y_pred, y = output
            y_pred = torch.sigmoid(y_pred)
            y_pred = torch.round(y_pred)
            return y_pred, y

        metrics = {
            "bin_accuracy": Accuracy(thresholded_output_transform),
            "loss": Loss(criterion)
        }
    else:
        criterion = nn.L1Loss()

        def bin_acc_transform(output):
            y_pred, y = output
            yp, yt = (y_pred > 0).long(), (y > 0).long()
            return yp, yt

        def mult_acc_transform(output):
            y_pred, y = output
            yp = torch.clamp(torch.round(y_pred) + 3, 0, 6).view(-1).long()
            yt = torch.round(y).view(-1).long() + 3
            yp = F.one_hot(yp, 7)
            return yp, yt

        metrics = {
            "bin_accuracy": Accuracy(output_transform=bin_acc_transform),
            "mae": MeanAbsoluteError(),
            #"cm": ConfusionMatrix(2, output_transform=bin_acc_transform),
            "mult_accuracy": Accuracy(output_transform=mult_acc_transform),
            "f1": Fbeta(1, output_transform=bin_acc_transform),
            "loss": Loss(criterion)
        }

    num_classes = 1
    model = AudioTextClassifier(
        audio_cfg=C["audio"]["model"],
        text_cfg=C["text"]["model"],
        fuse_cfg=C["fuse"],
        device=C["device"],
        modalities=C["modalities"],
        text_mode="glove",
        audio_mode="sequential"
    )

    optimizer = getattr(torch.optim, C['optimizer']['name'])(
        [p for p in model.parameters() if p.requires_grad],
        lr=C['optimizer']['learning_rate']
    )

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

    trainer.fit(
        train_loader,
        dev_loader,
        epochs=C['trainer']['max_epochs']
    )
    del trainer

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
        clip_grad_norm=C["trainer"]["clip_grad_norm"],
        loss_fn=criterion,
        device=C['device'],
    )

    testmetrics = trainer.predict(test_loader).metrics
    pprint(testmetrics)
