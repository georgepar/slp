import argparse
import glob
import os
import sys
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.metrics import Accuracy, Fbeta, Loss, MeanAbsoluteError
from torch.utils.data import DataLoader

from slp.config.nlp import SPECIAL_TOKENS
from slp.data.collators import IEMOCAPCollator
from slp.data.mosi import IEMOCAP_MULT
from slp.data.transforms import ToTensor, ToTokenIds
from slp.mm.load import data_pickle
# from slp.data.transforms import InstanceNorm, ToTokenIds, ToTensor, FilterCovarep
from slp.modules.mm import AudioVisualTextClassifier
from slp.modules.rnn import WordRNN
from slp.trainer import IEMOCAPTrainer
from slp.ui.config import load_config
from slp.util import log
from slp.util.system import safe_mkdirs


def get_parser():
    parser = argparse.ArgumentParser(description="CLI parser for experiment")

    parser.add_argument(
        "--binary",
        dest="binary",
        default=False,
        action="store_true",
        help="Use binary task",
    )

    parser.add_argument(
        "--remove-pauses",
        dest="remove_pauses",
        default=False,
        action="store_true",
        help="Remove speaker pauses",
    )

    parser.add_argument(
        "--modalities",
        dest="modalities",
        default=None,
        nargs="+",
        help="Modalities to use",
    )

    parser.add_argument(
        "--audio-dim",
        dest="audio.dim",
        default=None,
        type=int,
        help="Audio features dimension",
    )

    parser.add_argument(
        "--audio-instance-norm",
        dest="audio.instance_norm",
        action="store_true",
        help="Audio instance normalization",
    )

    parser.add_argument(
        "--audio-batchnorm",
        dest="audio.batchnorm",
        action="store_true",
        help="Audio batch normalization",
    )

    parser.add_argument(
        "--audio-layers",
        dest="audio.layers",
        default=None,
        type=int,
        help="Num layers for audio encoder",
    )

    parser.add_argument(
        "--text-layers",
        dest="text.layers",
        default=None,
        type=int,
        help="Num layers for text encoder",
    )

    parser.add_argument(
        "--audio-hidden-size",
        dest="audio.hidden_size",
        default=None,
        type=int,
        help="Hidden size for RNNs",
    )

    parser.add_argument(
        "--text-hidden-size",
        dest="text.hidden_size",
        default=None,
        type=int,
        help="Hidden size for RNNs",
    )

    parser.add_argument(
        "--dropout",
        dest="common.dropout",
        default=None,
        type=float,
        help="Dropout probabiity",
    )

    parser.add_argument(
        "--proj-size",
        dest="fuse.projection_size",
        default=None,
        type=int,
        help="Modality projection size",
    )

    parser.add_argument(
        "--bidirectional",
        dest="common.bidirectional",
        action="store_true",
        help="Use BiRNNs",
    )

    parser.add_argument(
        "--rnn-type", dest="common.rnn_type", default=None, type=str, help="lstm or gru"
    )

    parser.add_argument(
        "--text-attention",
        dest="text.attention",
        action="store_true",
        help="Use RNN with attention",
    )

    parser.add_argument(
        "--audio-attention",
        dest="audio.attention",
        action="store_true",
        help="Use RNN with attention",
    )

    parser.add_argument(
        "--fuse", dest="fuse.method", default=None, type=str, help="cat or add"
    )

    parser.add_argument(
        "--modality-weights",
        dest="fuse.modality_weights",
        action="store_true",
        help="Use modality weights during fusion",
    )

    parser.add_argument(
        "--feedback",
        dest="feedback",
        action="store_true",
        help="Use feedback fusion",
    )

    parser.add_argument(
        "--result-dir",
        dest="results_dir",
        help="Results directory",
    )

    return parser


C = load_config(parser=get_parser())
C["modalities"] = set(C["modalities"])

collate_fn = IEMOCAPCollator(
    device="cpu", binary=False, modalities=["text", "audio", "visual"], max_length=-1
)


if __name__ == "__main__":
    log.info("Running with configuration")
    pprint(C)
    train, dev, test, vocab = data_pickle(
        C["data_dir"],
    )

    to_tensor = ToTensor(device="cpu")
    to_tensor_float = ToTensor(device="cpu", dtype=torch.float)

    def create_dataloader(data, shuffle=False):
        d = IEMOCAP_MULT(data, modalities=C["modalities"])
        d.map(to_tensor_float, "visual", lazy=True)

        d.map(to_tensor_float, "text", lazy=True)

        d = d.map(to_tensor_float, "audio", lazy=True)
        d.apply_transforms()
        dataloader = DataLoader(
            d,
            batch_size=C["dataloaders"]["batch_size"],
            num_workers=C["dataloaders"]["num_workers"],
            pin_memory=C["dataloaders"]["batch_size"],
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

        return dataloader

    train_loader = create_dataloader(train, shuffle=True)
    dev_loader = create_dataloader(dev, shuffle=False)
    test_loader = create_dataloader(test, shuffle=False)

    # x = next(iter(train_loader))
    print("Running with feedback = {}".format(C["feedback"]))

    model = AudioVisualTextClassifier(
        embeddings=None,
        audio_cfg=C["audio"]["model"],
        text_cfg=C["text"]["model"],
        visual_cfg=C["visual"]["model"],
        fuse_cfg=C["fuse"],
        device=C["device"],
        modalities=C["modalities"],
        text_mode="glove",
        num_classes=8,
        feedback=C["feedback"],
    )
    model = model.to(C["device"])
    optimizer = getattr(torch.optim, C["optimizer"]["name"])(
        [p for p in model.parameters() if p.requires_grad],
        lr=C["optimizer"]["learning_rate"],
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=0.5,
        patience=2,
        cooldown=2,
        min_lr=C["optimizer"]["learning_rate"] / 20.0,
    )

    criterion = nn.CrossEntropyLoss()

    metrics = {
       "loss": Loss(criterion),
    }
    # score_fn = lambda engine: engine.state.metrics["bin_accuracy"]

    if C["overfit_batch"] or C["overfit_batch"] or C["train"]:
        import shutil
        try:
            shutil.rmtree(C["trainer"]["checkpoint_dir"])
        except:
            pass

        if C["trainer"]["accumulation_steps"] is not None:
            acc_steps = C["trainer"]["accumulation_steps"]
        else:
            acc_steps = 1
        trainer = IEMOCAPTrainer(
            model,
            optimizer,
            # score_fn=score_fn,
            experiment_name=C["experiment"]["name"],
            checkpoint_dir=C["trainer"]["checkpoint_dir"],
            metrics=metrics,
            non_blocking=C["trainer"]["non_blocking"],
            patience=C["trainer"]["patience"],
            validate_every=C["trainer"]["validate_every"],
            retain_graph=C["trainer"]["retain_graph"],
            loss_fn=criterion,
            accumulation_steps=acc_steps,
            lr_scheduler=lr_scheduler,
            device=C["device"],
        )

    if C["debug"]:
        if C["overfit_batch"]:
            trainer.overfit_single_batch(train_loader)
        trainer.fit_debug(train_loader, dev_loader)
        sys.exit(0)

    if C["train"]:
        trainer.fit(train_loader, dev_loader, epochs=C["trainer"]["max_epochs"])

    if C["test"]:
        try:
            del trainer
        except:
            pass
        trainer = IEMOCAPTrainer(
            model,
            optimizer,
            experiment_name=C["experiment"]["name"],
            checkpoint_dir=C["trainer"]["checkpoint_dir"],
            metrics=metrics,
            model_checkpoint=C["trainer"]["load_model"],
            non_blocking=C["trainer"]["non_blocking"],
            patience=C["trainer"]["patience"],
            validate_every=C["trainer"]["validate_every"],
            retain_graph=C["trainer"]["retain_graph"],
            loss_fn=criterion,
            device=C["device"],
        )

        predictions, targets = trainer.predict(test_loader)

        pred = torch.cat(predictions)
        y_test = torch.cat(targets)

        from slp.util.mosei_metrics import eval_iemocap, print_metrics, save_metrics
        import uuid

        metrics = eval_iemocap(pred, y_test)
        print_metrics(metrics)

        results_dir = C["results_dir"]
        safe_mkdirs(results_dir)
        fname = uuid.uuid1().hex
        results_file = os.path.join(results_dir, fname)

        save_metrics(metrics, results_file)
