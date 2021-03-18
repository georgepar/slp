from typing import Any, Callable, Optional

import torch
import torchmetrics
from torchmetrics.functional.classification.accuracy import _accuracy_compute


def _multiclass_accuracy_update(preds, targets):
    correct = torch.sum(torch.round(preds) == torch.round(targets))
    total = targets.size(0)

    return correct, total


class MoseiAcc2(torchmetrics.Metric):
    def __init__(
        self,
        exclude_non_zero: bool = False,
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
        self.exclude_non_zero = exclude_non_zero

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if self.exclude_non_zero:
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
