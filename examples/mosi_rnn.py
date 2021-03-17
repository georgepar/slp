import pytorch_lightning as pl
import torch.nn as nn
from loguru import logger
from slp.data.cmusdk import mosi
from slp.data.collators import MultimodalSequenceClassificationCollator
from slp.data.multimodal import MOSI
from slp.modules.classifier import MOSEITextClassifier, RNNLateFusionClassifier
from slp.plbind.dm import PLDataModuleFromDatasets
from slp.plbind.helpers import FromLogits
from slp.plbind.module import RnnPLModule
from slp.plbind.trainer import make_trainer, watch_model
from slp.util.log import configure_logging
from torch.optim import Adam

if __name__ == "__main__":
    EXPERIMENT_NAME = "mosi-rnn"

    configure_logging(f"logs/{EXPERIMENT_NAME}")

    modalities = {"text"}  # {"text", "audio", "visual"}
    collate_fn = MultimodalSequenceClassificationCollator(
        device="cpu", modalities=modalities
    )

    train_data, dev_data, test_data, w2v = mosi(
        "data/mosi_final_aligned",
        pad_back=False,
        max_length=-1,
        pad_front=True,
        remove_pauses=False,
        modalities=modalities,
        already_aligned=True,
        align_features=False,
        cache="./cache/mosi.p",
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

    train = MOSI(train_data, modalities=modalities, text_is_tokens=False)
    dev = MOSI(dev_data, modalities=modalities, text_is_tokens=False)
    test = MOSI(test_data, modalities=modalities, text_is_tokens=False)

    ldm = PLDataModuleFromDatasets(
        train,
        val=dev,
        test=test,
        batch_size=16,
        batch_size_eval=16,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0,
    )
    ldm.setup()

    model = MOSEITextClassifier(
        300,
        1,
        hidden_size=100,
        layers=1,
        attention=True,
        nystrom=False,
        kernel_size=None,
    )

    # feature_sizes = {"audio": 74, "visual": 35, "text": 300}

    # model = RNNLateFusionClassifier(
    #     feature_sizes,
    #     1,
    #     attention=True,
    #     nystrom=False,
    #     kernel_size=None,
    #     num_layers=1,
    #     num_heads=1,
    #     dropout=0.1,
    #     hidden_size=100,
    # )

    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    criterion = nn.MSELoss()

    lm = RnnPLModule(
        model,
        optimizer,
        criterion,
        # metrics={"acc": FromLogits(pl.metrics.classification.Accuracy())},
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
