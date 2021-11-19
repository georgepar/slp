import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from loguru import logger
from slp.data.cmusdk import mosei
from slp.data.collators import MultimodalSequenceClassificationCollator
from slp.data.multimodal import MOSEI
from slp.modules.multimodal import MultimodalBaselineClassifier
from slp.plbind.dm import PLDataModuleFromDatasets
from slp.plbind.helpers import FromLogits
from slp.plbind.metrics import MoseiAcc2, MoseiAcc5, MoseiAcc7
from slp.plbind.module import RnnPLModule
from slp.plbind.trainer import make_trainer, watch_model
from slp.util.log import configure_logging
from slp.util.mosei import run_evaluation
from torch.optim import Adam

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
        batch_size=16,
        batch_size_eval=16,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=2,
    )
    ldm.setup()

    model = MultimodalBaselineClassifier(1)
    print(model)
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2
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
        patience=10,
        gpus=1,
        save_top_k=1,
    )
    # watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)
    # trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())

    best_model = RnnPLModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=scheduler,
    )

    run_evaluation(best_model, ldm.test_dataloader(), "baseline_results4.csv")
