import argparse
import os

import torch
import torch.nn.functional as F

from collections import OrderedDict

from pprint import pprint
from torch.utils.data import DataLoader

from slp.config.nlp import SPECIAL_TOKENS
from slp.mm.load import mosi
from slp.data.mosi import MOSI
from slp.data.collators import MOSICollator
from slp.data.transforms import InstanceNorm, ToTokenIds, ToTensor
from slp.modules.multimodal import AudioTextClassifier, TextClassifier, AudioClassifier
from slp.ui.config import load_config
from slp.util.embeddings import EmbeddingsLoader
from slp.util import log
from slp.util import to_device

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)


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
        '--hidden-size', dest='common.hidden_size',
        default=None, type=int,
        help='Hidden size for RNNs')

    parser.add_argument(
        '--dropout', dest='common.dropout',
        default=None, type=float,
        help='Dropout probabiity')

    parser.add_argument(
        '--bidirectional', dest='common.bidirectional',
        action="store_true",
        help='Use BiRNNs')

    parser.add_argument(
        '--rnn-type', dest='common.rnn_type',
        default=None, type=str,
        help='lstm or gru')

    parser.add_argument(
        '--attention', dest='common.attention',
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


class MOSIPL(LightningModule):
    def __init__(self, C):
        super(MOSIPL, self).__init__()
        self.C = C
        self.hparams = obj(C)
        self.traindat, self.devdat, self.testdat, self.vocab = mosi(
            C['data_dir'],
            modalities=C['modalities'],
            remove_pauses=C['remove_pauses'],
            remove_neutral=C['binary'],
            cache=os.path.join(C['cache_dir'], 'mosi.p')
        )

        loader = EmbeddingsLoader(
            C['embeddings']['path'], C['embeddings']['dim'],
            extra_tokens=SPECIAL_TOKENS, vocab=self.vocab
        )
        self.word2idx, _, embeddings = loader.load()

        num_classes = 1 if C["binary"] else 5
        self.model = TextClassifier(
            C["audio"]["dim"], hidden_size=C["common"]["hidden_size"],
            embeddings=embeddings, vocab_size=len(self.word2idx),
            embeddings_dim=C["embeddings"]["dim"], batch_first=True,
            text_layers=C["text"]["layers"],
            audio_layers=C["audio"]["layers"],
            audio_batchnorm=C["audio"]["batchnorm"],
            embeddings_dropout=C["embeddings"]["dropout"],
            dropout=C["common"]["dropout"],
            finetune_embeddings=C["embeddings"]["finetune"],
            bidirectional=C["common"]["bidirectional"],
            rnn_type=C["common"]["rnn_type"],
            packed_sequence=True, attention=C["common"]["attention"],
            device=C["device"], fuse=C["fuse"]["method"],
            modality_weights=C["fuse"]["modality_weights"],
            return_hidden=False, num_classes=num_classes
        ).to(C['device'])

    def parse_batch(self, batch):
        batch = {
            m: to_device(v, device=self.C['device'], non_blocking=self.C['trainer']['non_blocking'])
            for m, v in batch.items()
        }
        return batch

    def __mkdataloader(self, data, train=True):
        to_token_ids = ToTokenIds(self.word2idx)
        to_tensor = ToTensor(device="cpu")
        to_tensor_float = ToTensor(device="cpu", dtype=torch.float)
        instance_norm = InstanceNorm()

        d = (
            MOSI(data, binary=self.C['binary'], modalities=self.C['modalities'])
            .map(to_token_ids, 'text', lazy=True)
            .map(to_tensor, 'text', lazy=True)
            #.map(instance_norm, 'visual', lazy=True)
            .map(to_tensor_float, 'visual', lazy=True)
        )
        if self.C["audio"]["instance_norm"]:
            d = d.map(instance_norm, "audio", lazy=True)
        d = d.map(to_tensor_float, 'audio', lazy=True)
        d.apply_transforms()
        shuffle = train
        return DataLoader(
            d,
            batch_size=self.C['dataloaders']['batch_size'],
            num_workers=self.C['dataloaders']['num_workers'],
            pin_memory=self.C['dataloaders']['batch_size'],
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        train_loader = self.__mkdataloader(self.traindat, train=True)
        return train_loader

    def val_dataloader(self):
        val_loader = self.__mkdataloader(self.devdat, train=False)
        return val_loader

    def test_dataloader(self):
        test_loader = self.__mkdataloader(self.testdat, train=False)
        return test_loader

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.C['optimizer']['name'])(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.C['optimizer']['learning_rate']
        )
        return optimizer

    def __acc(self, y_pred, y_true):
        preds = torch.round(torch.sigmoid(y_pred))
        acc = torch.sum(y_true == preds).item() / (len(y_true) * 1.0)
        return acc

    def training_step(self, batch, batch_idx):
        batch = self.parse_batch(batch)
        target = batch["labels"].view(-1, 1).float()
        y_pred = self(batch)
        loss = F.binary_cross_entropy_with_logits(y_pred, target)
        acc = self.__acc(y_pred, target)
        tqdm_dict = {'train_loss': loss}
        output = OrderedDict({
            'loss': loss,
            'acc': acc,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_idx):
        batch = self.parse_batch(batch)
        target = batch["labels"].view(-1, 1).float()
        y_pred = self(batch)
        loss = F.binary_cross_entropy_with_logits(y_pred, target)
        acc = self.__acc(y_pred, target)
        output = OrderedDict({
            'val_loss': loss,
            'val_acc': acc,
        })
        return output

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this during testing, similar to `validation_step`,
        with the data from the test dataloader passed in as `batch`.
        """
        output = self.validation_step(batch, batch_idx)
        # Rename output keys
        output['test_loss'] = output.pop('val_loss')
        output['test_acc'] = output.pop('val_acc')

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}
        for metric_name in ["val_loss", "val_acc"]:
            metric_total = 0
            for output in outputs:
                metric_value = output[metric_name]
                # reduce manually when using dp
                if self.trainer.use_dp or self.trainer.use_ddp2:
                    metric_value = torch.mean(metric_value)
                metric_total += metric_value
            tqdm_dict[metric_name] = metric_total / len(outputs)
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"]}
        return result

    def test_epoch_end(self, outputs):
        """
        Called at the end of test to aggregate outputs, similar to `validation_epoch_end`.
        :param outputs: list of individual outputs of each test step
        """
        import ipdb; ipdb.set_trace()
        results = self.validation_step_end(outputs)

        # rename some keys
        results['progress_bar'].update({
            'test_loss': results['progress_bar'].pop('val_loss'),
            'test_acc': results['progress_bar'].pop('val_acc'),
        })
        results['log'] = results['progress_bar']
        results['test_loss'] = results.pop('val_loss')

        return results



if __name__ == "__main__":
    log.info('Running with configuration')
    pprint(C)

    model = MOSIPL(C)
    # DEFAULTS used by the Trainer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=C['trainer']['checkpoint_dir'],
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=C['experiment']['name']
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=C['trainer']['patience'],
        verbose=False,
        mode='min'
    )
    trainer = pl.Trainer(
        default_root_dir=C['trainer']['checkpoint_dir'],
        gpus=0 if C['device'] == 'cuda' else C['device'],
        max_epochs=C['trainer']['max_epochs'],
        check_val_every_n_epoch=C['trainer']['validate_every'],
        checkpoint_callback=checkpoint_callback
    )

    if C['train']:
        trainer.fit(model)

    if C['test']:
        trainer.test()
        import ipdb; ipdb.set_trace()
