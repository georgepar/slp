import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from loguru import logger
from slp.data.cmusdk import mosei
from slp.data.collators import MultimodalSequenceClassificationCollator
from slp.data.multimodal import MOSEI
from slp.modules.classifier import MOSEITextClassifier, RNNLateFusionClassifier
from slp.plbind.dm import PLDataModuleFromDatasets
from slp.plbind.helpers import FromLogits
from slp.plbind.metrics import MoseiAcc2, MoseiAcc5, MoseiAcc7
from slp.plbind.module import RnnPLModule
from slp.plbind.trainer import make_trainer, watch_model
from slp.util.log import configure_logging
from torch.optim import AdamW

if __name__ == "__main__":
    EXPERIMENT_NAME = "mosei-rnn"

    configure_logging(f"logs/{EXPERIMENT_NAME}")

    modalities = {"text", "audio", "visual"}
    collate_fn = MultimodalSequenceClassificationCollator(
        device="cpu", modalities=modalities
    )

    train_data, dev_data, test_data, w2v = mosei(
        "data/mosei_final_aligned",
        pad_back=False,
        max_length=-1,
        pad_front=True,
        remove_pauses=False,
        modalities=modalities,
        already_aligned=True,
        align_features=False,
        cache="./cache/mosei.p",
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

    train = MOSEI(train_data, modalities=modalities, text_is_tokens=False)
    dev = MOSEI(dev_data, modalities=modalities, text_is_tokens=False)
    test = MOSEI(test_data, modalities=modalities, text_is_tokens=False)

    ldm = PLDataModuleFromDatasets(
        train,
        val=dev,
        test=test,
        batch_size=8,
        batch_size_eval=8,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=2,
    )
    ldm.setup()

    feature_sizes = {"audio": 74, "visual": 35, "text": 300}

    model = RNNLateFusionClassifier(
        feature_sizes,
        1,
        attention=True,
        nystrom=False,
        kernel_size=None,
        num_layers=1,
        num_heads=1,
        dropout=0.1,
        hidden_size=100,
        use_mmdrop=True,
        mmdrop_mode="soft",
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=5
    )

    criterion = nn.L1Loss()

    lm = RnnPLModule(
        model,
        optimizer,
        criterion,
        lr_scheduler=scheduler,
        metrics={
            "acc2": MoseiAcc2(exclude_neutral=True),
            "acc2_zero": MoseiAcc2(exclude_neutral=False),
            "acc5": MoseiAcc5(),
            "acc7": MoseiAcc7(),
            "mae": torchmetrics.MeanAbsoluteError(),
        },
    )

    trainer = make_trainer(
        EXPERIMENT_NAME,
        max_epochs=100,
        gpus=1,
        save_top_k=1,
    )
    watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)

    trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())
