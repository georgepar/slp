import argparse
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
from slp.data.collators import MOSICollator
from slp.data.mosi import MOSEI
from slp.data.transforms import ToTensor, ToTokenIds
from slp.mm.load import mosei
# from slp.data.transforms import InstanceNorm, ToTokenIds, ToTensor, FilterCovarep
from slp.modules.multimodal import AudioVisualTextTransformerClassifier
from slp.trainer import MOSITrainer
from slp.ui.config import load_config
from slp.util import log
from slp.util.system import safe_mkdirs

#torch.autograd.set_detect_anomaly(True)

class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()

    def forward(self, out, tgt):
        tgt = tgt.view(-1, 1).float()

        return F.binary_cross_entropy_with_logits(out, tgt)


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
        dest="transformer.audio_size",
        default=None,
        type=int,
        help="Audio features dimension",
    )

    parser.add_argument(
        "--text-dim",
        dest="transformer.text_size",
        default=None,
        type=int,
        help="Text features dimension",
    )

    parser.add_argument(
        "--visual-dim",
        dest="transformer.visual_size",
        default=None,
        type=int,
        help="Visual features dimension",
    )

    parser.add_argument(
        "--hidden-size",
        dest="transformer.hidden_size",
        default=None,
        type=int,
        help="Hidden size",
    )

    parser.add_argument(
        "--layers",
        dest="transformer.num_layers",
        default=None,
        type=int,
        help="Num layers",
    )

    parser.add_argument(
        "--num-heads",
        dest="transformer.num_heads",
        default=None,
        type=int,
        help="Num heads",
    )

    parser.add_argument(
        "--max-length",
        dest="transformer.max_length",
        default=None,
        type=int,
        help="Max length",
    )

    parser.add_argument(
        "--inner-size",
        dest="transformer.inner_size",
        default=None,
        type=int,
        help="Num heads",
    )

    parser.add_argument(
        "--dropout",
        dest="transformer.dropout",
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

collate_fn = MOSICollator(
    device="cpu", binary=False, modalities=["text", "audio", "visual"], max_length=-1
)


if __name__ == "__main__":
    log.info("Running with configuration")
    pprint(C)
    train, dev, test, vocab = mosei(
        C["data_dir"],
        modalities=C["modalities"],
        remove_pauses=C['remove_pauses'],
        max_length=-1,
        pad_front=False,
        pad_back=False,
        aligned=False,
        cache=os.path.join(C["cache_dir"], "mosei_avt_unpadded.p"),
    )

    assert "glove" in C["modalities"], "Use glove"

    if "glove" in C["modalities"]:
        for d in train:
            d["text"] = d["glove"]

        for d in dev:
            d["text"] = d["glove"]

        for d in test:
            d["text"] = d["glove"]


    from tqdm import tqdm
    # normalize
    all_audio = []
    for d in tqdm(train):
        x = d["audio"]
        all_audio.append(x)

    all_audio = np.vstack(all_audio).astype(np.float64)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(all_audio)

    import pickle

    with open("cache/covarep_scaler.p", "wb") as fd:
        pickle.dump(scaler, fd)

    del all_audio

    for d in tqdm(train):
        d["audio"] = scaler.transform(d["audio"])

    for d in tqdm(dev):
        d["audio"] = scaler.transform(d["audio"])

    for d in tqdm(test):
        d["audio"] = scaler.transform(d["audio"])



    to_tensor_float = ToTensor(device="cpu", dtype=torch.float)

    def create_dataloader(data):
        d = (
            MOSEI(data, modalities=C["modalities"], unpad=False, select_label=0)
        )
        d.map(to_tensor_float, "visual", lazy=True)
        d.map(to_tensor_float, "text", lazy=True)
        d = d.map(to_tensor_float, "audio", lazy=True)
        d.apply_transforms()
        dataloader = DataLoader(
            d,
            batch_size=C["dataloaders"]["batch_size"],
            num_workers=C['dataloaders']['num_workers'],
            pin_memory=C["dataloaders"]["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
        )

        return dataloader

    train_loader, dev_loader, test_loader = map(create_dataloader, [train, dev, test])
    # x = next(iter(train_loader))
    print("Running with feedback = {}".format(C["feedback"]))

    model = AudioVisualTextTransformerClassifier(
        transformer_cfg=C["transformer"],
        fuse_cfg=C["fuse"],
        device=C["device"],
        num_classes=1,
        feedback=C["feedback"],
    )
    model = model.to(C["device"])
    optimizer = getattr(torch.optim, C["optimizer"]["name"])(
        [p for p in model.parameters() if p.requires_grad],
        lr=C["optimizer"]["learning_rate"],
    )

    if C["binary"]:
        criterion = BCE()

        def thresholded_output_transform(output):
            y_pred, y = output
            y_pred = torch.sigmoid(y_pred)
            y_pred = torch.round(y_pred)

            return y_pred, y

        metrics = {
            "bin_accuracy": Accuracy(thresholded_output_transform),
            "loss": Loss(criterion),
        }
        # score_fn = lambda engine: engine.state.metrics["bin_accuracy"]
    else:
        criterion = nn.L1Loss()

        def bin_acc_transform(output):
            y_pred, y = output
            yp, yt = (y_pred > 0).long(), (y > 0).long()

            return yp, yt

        metrics = {
            # "bin_accuracy": Accuracy(output_transform=bin_acc_transform),
            "loss": Loss(criterion),
        }
        # score_fn = lambda engine: engine.state.metrics["bin_accuracy"]

    if C["overfit_batch"] or C["overfit_batch"] or C["train"]:
        trainer = MOSITrainer(
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
            device=C["device"],
            # parallel=True
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
        trainer = MOSITrainer(
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

        from slp.util.mosei_metrics import eval_mosei_senti, print_metrics, save_metrics
        import uuid

        metrics = eval_mosei_senti(pred, y_test, True)
        print_metrics(metrics)

        results_dir = C["results_dir"]
        safe_mkdirs(results_dir)
        fname = uuid.uuid1().hex
        results_file = os.path.join(results_dir, fname)

        save_metrics(metrics, results_file)


        metrics = eval_mosei_senti(pred, y_test, False)

        results_dir = C["results_dir"] + "_neutral"
        safe_mkdirs(results_dir)
        results_file = os.path.join(results_dir, fname)

        save_metrics(metrics, results_file)


