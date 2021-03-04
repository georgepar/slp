import torch

from typing import List, Tuple, cast
from slp.util.types import Label
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from slp.util.pytorch import mktensor, pad_mask, subsequent_mask


class SequenceClassificationCollator(object):
    def __init__(self, pad_indx=0, device="cpu"):
        """Collate function for sequence classification tasks

        * Perform padding
        * Calculate sequence lengths

        Args:
            pad_indx (int): Pad token index. Defaults to 0.
            device (str): device of returned tensors. Leave this as "cpu".
                The LightningModule will handle the Conversion.

        Examples:
        >>> dataloader = torch.utils.DataLoader(my_dataset, collate_fn=SequenceClassificationCollator())
        """
        self.pad_indx = pad_indx
        self.device = device

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
        # Pad and convert to tensor
        inputs_padded: torch.Tensor = pad_sequence(
            inputs, batch_first=True, padding_value=self.pad_indx
        ).to(self.device)
        ttargets: torch.Tensor = mktensor(targets, device=self.device, dtype=torch.long)
        return inputs_padded, ttargets.to(self.device), lengths


class Seq2SeqCollator(object):
    def __init__(self, pad_indx=0, device="cpu"):
        """Collate function for seq2seq tasks

        * Perform padding
        * Calculate sequence lengths

        Args:
            pad_indx (int): Pad token index. Defaults to 0.
            device (str): device of returned tensors. Leave this as "cpu".
                The LightningModule will handle the Conversion.

        Examples:
        >>> dataloader = torch.utils.DataLoader(my_dataset, collate_fn=Seq2SeqClassificationCollator())
        """
        self.pad_indx = pad_indx
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

        inputs_padded: torch.Tensor = pad_sequence(
            inputs, batch_first=True, padding_value=self.pad_indx
        ).to(self.device)

        targets_padded: torch.Tensor = pad_sequence(
            targets, batch_first=True, padding_value=self.pad_indx
        ).to(self.device)

        return inputs_padded, targets_padded, lengths_inputs, lengths_targets
