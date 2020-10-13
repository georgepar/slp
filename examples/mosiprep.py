import pickle
import numpy as np
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.metrics import Loss, Accuracy, Fbeta, MeanAbsoluteError
from pprint import pprint
from torch.utils.data import DataLoader

from slp.data.mosi import MOSIMFN
from slp.data.collators import MOSICollator
from slp.data.transforms import InstanceNorm, ToTensor
from slp.modules.multimodal import AudioTextClassifier
from slp.trainer import MOSITrainer
from slp.ui.config import load_config
from slp.util import log
from slp.util import split


def load_saved_data(path, lmf=True):
    if lmf:
        with open(os.path.join(path, 'mosi_data.pkl'), 'rb') as fd:
            data = pickle.load(fd)

        train, dev, test = data['train'], data['valid'], data['test']
        X_train = np.concatenate((train['text'], train['audio'], train['vision']), axis=2)
        X_dev = np.concatenate((dev['text'], dev['audio'], dev['vision']), axis=2)
        X_test = np.concatenate((test['text'], test['audio'], test['vision']), axis=2)
        y_train = train['labels'].flatten()
        y_dev = dev['labels'].flatten()
        y_test = test['labels'].flatten()

        X = np.concatenate((X_train, X_dev, X_test), axis=0)
        vi = X[:, :, 305:]
        vi_max = np.max(np.max(np.abs(vi), axis=0), axis=0)
        vi_max[vi_max == 0] = 1
        vi = vi / vi_max
        X[:, :, 305:] = vi
        y = np.concatenate((y_train, y_dev, y_test))
    else:
        with open(os.path.join(path, 'mosi_efthymis.pkl'), 'rb') as fd:
            data = pickle.load(fd)
        X, y = [], []
        for el in data:
            xx = np.concatenate((el['text'], el['audio']), axis=1)
            X.append(xx)
            y.append(el['label'].item())
    X = np.array(X)
    X = np.nan_to_num(X)
    y = np.array(y)
    y = np.nan_to_num(y)
    return X, y


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
    binary=C['binary'],
    modalities=C['modalities'],
    max_length=-1
)


if __name__ == "__main__":
    log.info('Running with configuration')

    pprint(C)

    X, y = load_saved_data(C['data_dir'], lmf=True)

    to_tensor = ToTensor(device="cpu")
    to_tensor_float = ToTensor(device="cpu", dtype=torch.float)
    instance_norm = InstanceNorm()

    d = (
        MOSIMFN(
            X, y, binary=C['binary'], unpad=False,
            modalities=C['modalities'], lmf=True
        )
        .map(to_tensor_float, 'text', lazy=True)
    )
    if C["audio"]["instance_norm"]:
        d = d.map(instance_norm, "audio", lazy=True)
        d = d.map(instance_norm, 'visual', lazy=True)
    d = d.map(to_tensor_float, 'audio', lazy=True)
    d = d.map(to_tensor_float, 'visual', lazy=True)
    d.apply_transforms()


    def create_dataloader(dataset, sampler=None):
        shuffle = True if sampler is None else False
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=C['dataloaders']['batch_size'],
            num_workers=C['dataloaders']['num_workers'],
            pin_memory=C['dataloaders']['batch_size'],
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

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
            "mult_accuracy": Accuracy(output_transform=mult_acc_transform),
            "f1": Fbeta(1, output_transform=bin_acc_transform),
            "loss": Loss(criterion)
        }

    kfold = split.kfold_split(d, create_dataloader, k=5)
    num_classes = 1
    kfold_metrics = {k: [] for k in metrics.keys()}
    for train_loader, dev_loader in kfold:
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
        valmetrics = trainer.predict(dev_loader).metrics
        pprint(valmetrics)
        for k in valmetrics.keys():
            kfold_metrics[k].append(valmetrics[k])
        del trainer
        del model

    for k, v in kfold_metrics.items():
       print("Kfold {}: {}+-{}".format(k, np.mean(v), np.std(v)))
