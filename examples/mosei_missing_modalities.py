import csv
import os
import sys

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from loguru import logger
from slp.config.config_parser import make_cli_parser, parse_config
from slp.data.cmusdk import mosei
from slp.data.collators import MultimodalSequenceClassificationCollator
from slp.data.multimodal import MOSEI
from slp.modules.baseline import AudioVisualTextClassifier
from slp.modules.classifier import RNNSymAttnFusionRNNClassifier
from slp.plbind.dm import PLDataModuleFromDatasets
from slp.plbind.helpers import FromLogits
from slp.plbind.metrics import MoseiAcc2, MoseiAcc5, MoseiAcc7, MoseiF1
from slp.plbind.module import RnnPLModule
from slp.plbind.trainer import make_trainer, watch_model
from slp.util.log import configure_logging
from slp.util.mosei import get_mosei_parser, mosei_run_test, patch_mosei_pickle
from slp.util.system import is_file, safe_mkdirs
from torch.optim import Adam
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # parser
    parser = get_mosei_parser()
    parser.add_argument("--ckpt", dest="ckpt_path", type=str, help="Checkpoint to test")
    parser.add_argument(
        "--missing-modalities",
        dest="missing_modalities",
        action="store_true",
        help="Test with missing modalities",
    )

    parser.add_argument(
        "--frame-drop-percentage",
        dest="frame_drop_percentage",
        type=float,
        default=1.0,
        help="Frame drop percentage",
    )

    parser.add_argument(
        "--modality-to-miss",
        dest="modality_to_miss",
        type=str,
        choices=["text", "audio", "visual"],
        default=None,
        help="Missing frame for a specific modality",
    )

    parser.add_argument(
        "--repeat-frame",
        dest="repeat_frame",
        action="store_true",
        help="Repeat previous frame when dropping frames",
    )

    parser = make_cli_parser(parser, PLDataModuleFromDatasets)

    config = parse_config(parser, parser.parse_args().config)

    configure_logging(f"logs/{config.trainer.experiment_name}")
    modalities = set(config.modalities)
    max_length = config.model.max_length
    collate_fn = MultimodalSequenceClassificationCollator(
        device="cpu", modalities=modalities
    )

    train_data, dev_data, test_data, w2v = mosei(
        "data/mosei_final_aligned/",
        modalities=modalities,
        max_length=-1,
        pad_back=config.preprocessing.pad_back,
        pad_front=config.preprocessing.pad_front,
        remove_pauses=config.preprocessing.remove_pauses,
        already_aligned=config.preprocessing.already_aligned,
        align_features=config.preprocessing.align_features,
        cache="./cache/mosei_avt_unpadded.p",
    )

    train_data, dev_data, test_data = patch_mosei_pickle(
        train_data, dev_data, test_data
    )

    for x in train_data:
        if "glove" in x:
            x["text"] = x["glove"]

    for x in dev_data:
        if "glove" in x:
            x["text"] = x["glove"]

    for x in test_data:
        if "glove" in x:
            x["text"] = x["glove"]

    test = MOSEI(
        test_data,
        modalities=modalities,
        text_is_tokens=False,
        missing_modalities=config.missing_modalities,
        modality_to_miss=config.modality_to_miss,
        frame_drop_percentage=config.frame_drop_percentage,
        repeat_frame=config.repeat_frame,
    )

    dataloader = DataLoader(
        test,
        batch_size=config.data.batch_size,
        shuffle=False,  # config.data.shuffle_eval,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=config.data.drop_last,
        collate_fn=collate_fn,
    )

    feature_sizes = config.model.feature_sizes

    model = AudioVisualTextClassifier(
        feature_sizes,
        1,
        num_layers=config.model.num_layers,
        batch_first=config.model.batch_first,
        bidirectional=config.model.bidirectional,
        packed_sequence=config.model.packed_sequence,
        merge_bi=config.model.merge_bi,
        rnn_type=config.model.rnn_type,
        attention=config.model.attention,
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        max_length=config.model.max_length,
        dropout=config.model.dropout,
        nystrom=False,
        multi_modal_drop=config.model.multi_modal_drop,
        p_mmdrop=config.model.p_mmdrop,
        mmdrop_before_fuse=config.model.mmdrop_before_fuse,
        mmdrop_after_fuse=config.model.mmdrop_after_fuse,
        p_drop_modalities=config.model.p_drop_modalities,
    )

    print(model)

    lm = RnnPLModule(
        model,
        Adam(model.parameters()),
        nn.L1Loss(),
        metrics={
            "acc2": MoseiAcc2(exclude_neutral=True),
            "acc2_zero": MoseiAcc2(exclude_neutral=False),
            "acc5": MoseiAcc5(),
            "acc7": MoseiAcc7(),
            "f1": MoseiF1(exclude_neutral=True),
            "f1_zero": MoseiF1(exclude_neutral=False),
            "mae": torchmetrics.MeanAbsoluteError(),
        },
    )

    results = mosei_run_test(lm, dataloader, config.ckpt_path, modalities)
    import pprint

    pprint.pprint(results)
    csv_folder_path = os.path.join(
        config.trainer.experiments_folder, config.trainer.experiment_name, "results_csv"
    )

    csv_name = os.path.join(csv_folder_path, "results.csv")
    fieldnames = list(results.keys())

    if is_file(csv_name):
        # folder already exits and so does the .csv
        csv_exists = True
        print(f"csv already exists")
    else:
        csv_exists = False
        safe_mkdirs(csv_folder_path)

    with open(csv_name, "a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not csv_exists:
            writer.writeheader()
        writer.writerow(results)
