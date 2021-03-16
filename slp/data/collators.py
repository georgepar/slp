from typing import Dict, List, Tuple

import torch

from slp.util.pytorch import mktensor, pad_sequence
from slp.util.types import Label


class SequenceClassificationCollator(object):
    def __init__(self, pad_indx=0, max_length=-1, device="cpu"):
        """Collate function for sequence classification tasks

        * Perform padding
        * Calculate sequence lengths

        Args:
            pad_indx (int): Pad token index. Defaults to 0.
            max_length (int): Pad sequences to a fixed maximum length
            device (str): device of returned tensors. Leave this as "cpu".
                The LightningModule will handle the Conversion.

        Examples:
            >>> dataloader = torch.utils.DataLoader(my_dataset, collate_fn=SequenceClassificationCollator())
        """
        self.pad_indx = pad_indx
        self.device = device
        self.max_length = max_length

    def __call__(
        self, batch: List[Tuple[torch.Tensor, Label]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Call collate function

        Args:
            batch (List[Tuple[torch.Tensor, slp.util.types.Label]]): Batch of samples.
                It expects a list of tuples (inputs, label).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Returns tuple of batched tensors (inputs, labels, lengths)
        """
        inputs: List[torch.Tensor] = [b[0] for b in batch]
        targets: List[Label] = [b[1] for b in batch]
        #  targets: List[torch.tensor] = map(list, zip(*batch))
        lengths = torch.tensor([s.size(0) for s in inputs], device=self.device)

        if self.max_length > 0:
            lengths = torch.clamp(lengths, min=0, max=self.max_length)
        # Pad and convert to tensor
        inputs_padded: torch.Tensor = pad_sequence(
            inputs,
            batch_first=True,
            padding_value=self.pad_indx,
            max_length=self.max_length,
        ).to(self.device)

        ttargets: torch.Tensor = mktensor(targets, device=self.device, dtype=torch.long)

        return inputs_padded, ttargets.to(self.device), lengths


class Seq2SeqCollator(object):
    def __init__(self, pad_indx=0, max_length=-1, device="cpu"):
        """Collate function for seq2seq tasks

        * Perform padding
        * Calculate sequence lengths

        Args:
            pad_indx (int): Pad token index. Defaults to 0.
            max_length (int): Pad sequences to a fixed maximum length
            device (str): device of returned tensors. Leave this as "cpu".
                The LightningModule will handle the Conversion.

        Examples:
            >>> dataloader = torch.utils.DataLoader(my_dataset, collate_fn=Seq2SeqClassificationCollator())
        """
        self.pad_indx = pad_indx
        self.max_length = max_length
        self.device = device

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Call collate function

        Args:
            batch (List[Tuple[torch.Tensor, torch.Tensor]]): Batch of samples.
                It expects a list of tuples (source, target)
                Each source and target are a sequences of features or ids.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Returns tuple of batched tensors
                (inputs, labels, lengths_inputs, lengths_targets)
        """
        inputs: List[torch.Tensor] = [b[0] for b in batch]
        targets: List[torch.Tensor] = [b[1] for b in batch]
        lengths_inputs = torch.tensor([s.size(0) for s in inputs], device=self.device)
        lengths_targets = torch.tensor([s.size(0) for s in targets], device=self.device)

        if self.max_length > 0:
            lengths_inputs = torch.clamp(lengths_inputs, min=0, max=self.max_length)
            lengths_targets = torch.clamp(lengths_targets, min=0, max=self.max_length)

        inputs_padded: torch.Tensor = pad_sequence(
            inputs,
            batch_first=True,
            padding_value=self.pad_indx,
            max_length=self.max_length,
        ).to(self.device)

        targets_padded: torch.Tensor = pad_sequence(
            targets,
            batch_first=True,
            padding_value=self.pad_indx,
            max_length=self.max_length,
        ).to(self.device)

        return inputs_padded, targets_padded, lengths_inputs, lengths_targets


class MultimodalSequenceClassificationCollator(object):
    def __init__(
        self,
        pad_indx=0,
        modalities={"visual", "text", "audio"},
        label_key="label",
        max_length=-1,
        label_dtype=torch.float,
        device="cpu",
    ):
        """Collate function for sequence classification tasks

        * Perform padding
        * Calculate sequence lengths

        Args:
            pad_indx (int): Pad token index. Defaults to 0.
            modalities (Set): Which modalities are included in the batch dict
            max_length (int): Pad sequences to a fixed maximum length
            label_key (str): String to access the label in the batch dict
            device (str): device of returned tensors. Leave this as "cpu".
                The LightningModule will handle the Conversion.

        Examples:
            >>> dataloader = torch.utils.DataLoader(my_dataset, collate_fn=MultimodalSequenceClassificationCollator())
        """
        self.pad_indx = pad_indx
        self.device = device
        self.max_length = max_length
        self.label_key = label_key
        self.modalities = modalities
        self.label_dtype = label_dtype

    def extract_sequence(self, batch, key) -> List[torch.Tensor]:
        return [b[key] for b in batch]

    def __call__(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """Call collate function

        Args:
            batch (List[Dict[str, torch.Tensor]]): Batch of samples.
                It expects a list of dictionaries from modalities to torch tensors

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]: tuple of
                (dict batched modality tensors, labels, dict of modality sequence lengths)
        """
        inputs = {}
        lengths = {}

        for m in self.modalities:
            seq = self.extract_sequence(batch, m)
            lengths[m] = torch.tensor([s.size(0) for s in seq], device=self.device)

            if self.max_length > 0:
                lengths[m] = torch.clamp(lengths[m], min=0, max=self.max_length)

            inputs[m] = pad_sequence(
                seq,
                batch_first=True,
                padding_value=self.pad_indx,
                max_length=self.max_length,
            ).to(self.device)

        targets: List[Label] = [b[self.label_key] for b in batch]

        # Pad and convert to tensor
        ttargets: torch.Tensor = mktensor(
            targets, device=self.device, dtype=self.label_dtype
        )

        return inputs, ttargets.to(self.device), lengths
