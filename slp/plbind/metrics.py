from typing import Any, Callable, Optional

import torch
import torchmetrics
from torchmetrics.functional.classification.accuracy import _accuracy_compute
from torchmetrics.functional.classification.f_beta import _fbeta_compute
from torchmetrics.utilities.data import to_onehot


def _multiclass_accuracy_update(preds, targets):
    correct = torch.sum(torch.round(preds) == torch.round(targets))
    total = targets.size(0)

    return correct, total


class MoseiAcc2(torchmetrics.Metric):
    def __init__(
        self,
        exclude_neutral: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super(MoseiAcc2, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.exclude_neutral = exclude_neutral

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if self.exclude_neutral:
            pp = (preds[targets != 0] >= 0).int()
            tt = (targets[targets != 0] >= 0).int()
        else:
            pp = (preds >= 0).int()
            tt = (targets >= 0).int()

        correct = (pp == tt).sum()
        total = tt.size(0)

        self.correct += correct
        self.total += total

    def compute(self):
        return _accuracy_compute(self.correct, self.total)


class MoseiMulticlassAcc(torchmetrics.Metric):
    def __init__(
        self,
        clamp=2,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super(MoseiMulticlassAcc, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.clamp = clamp
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        correct, total = _multiclass_accuracy_update(
            torch.clamp(preds, min=-self.clamp, max=self.clamp),
            torch.clamp(targets, min=-self.clamp, max=self.clamp),
        )

        self.correct += correct
        self.total += total

    def compute(self):
        return _accuracy_compute(self.correct, self.total)


class MoseiF1(torchmetrics.Metric):
    def __init__(
        self,
        exclude_neutral: bool = True,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super(MoseiF1, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.exclude_neutral = exclude_neutral
        self.add_state("true_positives", default=torch.zeros(2), dist_reduce_fx="sum")
        self.add_state(
            "predicted_positives", default=torch.zeros(2), dist_reduce_fx="sum"
        )
        self.add_state("actual_positives", default=torch.zeros(2), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if self.exclude_neutral:
            pp = (preds[targets != 0] >= 0).int()
            tt = (targets[targets != 0] >= 0).int()
        else:
            pp = (preds >= 0).int()
            tt = (targets >= 0).int()

        pp = to_onehot(pp, num_classes=2).transpose(1, 0).reshape(2, -1)
        tt = to_onehot(tt, num_classes=2).transpose(1, 0).reshape(2, -1)

        true_positives = torch.sum(pp * tt, dim=1)
        predicted_positives = torch.sum(pp, dim=1)
        actual_positives = torch.sum(tt, dim=1)

        self.true_positives += true_positives
        self.predicted_positives += predicted_positives
        self.actual_positives += actual_positives

    def compute(self):
        return _fbeta_compute(
            self.true_positives,
            self.predicted_positives,
            self.actual_positives,
            beta=1.0,
            average="weighted",
        )


class MoseiAcc5(MoseiMulticlassAcc):
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super(MoseiAcc5, self).__init__(
            clamp=2,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )


class MoseiAcc7(MoseiMulticlassAcc):
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super(MoseiAcc7, self).__init__(
            clamp=3,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
