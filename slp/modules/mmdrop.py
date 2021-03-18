import random
from typing import List, Optional

import torch
import torch.nn as nn


class HardMultimodalDropout(nn.Module):
    def __init__(
        self, p: float = 0.5, n_modalities: int = 3, p_mod: Optional[List[float]] = None
    ):
        """MMDrop initial implementation

        For each sample in a batch drop one of the modalities with probability p

        Args:
            p (float): drop probability
            n_modalities (int): number of modalities
            p_mod (Optional[List[float]]): Drop probabilities for each modality
        """
        super(HardMultimodalDropout, self).__init__()
        self.p = p
        self.n_modalities = n_modalities

        self.p_mod = [1.0 / n_modalities for _ in range(n_modalities)]

        if p_mod is not None:
            self.p_mod = p_mod

    def forward(self, *mods):
        """Naive mmdrop forward

        Iterate over batch and randomly choose modality to drop

        Args:
            mods (varargs torch.Tensor): [B, L, D_m] Modality representations

        Returns:
            (List[torch.Tensor]): The modality representations. Some of them are dropped
        """
        mods = list(mods)

        # List of [B, L, D]

        if self.training:
            if random.random() < self.p:
                # Drop different modality for each sample in batch

                for batch in range(mods[0].size(0)):
                    m = random.choices(
                        list(range(self.n_modalities)), weights=self.p_mod, k=1
                    )[0]

                    # m = random.randint(0, self.n_modalities - 1)
                    mask = torch.ones_like(mods[m])
                    mask[batch] = 0.0
                    mods[m] = mods[m] * mask

            if self.p > 0:
                for m in range(len(mods)):
                    keep_prob = 1 - (self.p / self.n_modalities)
                    mods[m] = mods[m] * (1 / keep_prob)

        return mods


class SoftMultimodalDropout(nn.Module):
    def __init__(
        self, p: float = 0.5, n_modalities: int = 3, p_mod: Optional[List[float]] = None
    ):
        """Soft mmdrop implementation

        Drop p * 100 % of features of a specific modality over batch

        Args:
            p (float): drop probability
            n_modalities (int): number of modalities
            p_mod (Optional[List[float]]): Drop probabilities for each modality
        """
        super(SoftMultimodalDropout, self).__init__()
        self.p = p  # p_drop
        self.n_modalities = n_modalities

        self.p_mod = [1.0 / n_modalities for _ in range(n_modalities)]

        if p_mod is not None:
            self.p_mod = p_mod

    def forward(self, *mods):
        """Soft mmdrop forward

        Sample a binomial mask to mask a random modality in this batch

        Args:
            mods (varargs torch.Tensor): [B, L, D_m] Modality representations

        Returns:
            (List[torch.Tensor]): The modality representations. Some of them are dropped
        """
        mods = list(mods)

        if self.training:
            # m = random.randint(0, self.n_modalities - 1)
            m = random.choices(list(range(self.n_modalities)), weights=self.p_mod, k=1)[
                0
            ]

            binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
            mods[m] = mods[m] * binomial.sample(mods[m].size()).to(mods[m].device)

            for m in range(self.n_modalities):
                mods[m] = mods[m] * (1.0 / (1 - self.p / self.n_modalities))

        return mods


class MultimodalDropout(nn.Module):
    def __init__(
        self,
        p: float = 0.5,
        n_modalities: int = 3,
        p_mod: Optional[List[float]] = None,
        mode: str = "hard",
    ):
        """mmdrop wrapper class

        Drop p * 100 % of features of a specific modality over batch

        Args:
            p (float): drop probability
            n_modalities (int): number of modalities
            p_mod (Optional[List[float]]): Drop probabilities for each modality
            mode (str): Hard or soft mmdrop
        """
        super(MultimodalDropout, self).__init__()

        assert mode in [
            "hard",
            "soft",
        ], "Allowed mode for MultimodalDropout ['hard' | 'soft']"

        if mode == "hard":
            self.mmdrop = HardMultimodalDropout(
                p=p, n_modalities=n_modalities, p_mod=p_mod
            )
        else:
            self.mmdrop = SoftMultimodalDropout(  # type: ignore
                p=p, n_modalities=n_modalities, p_mod=p_mod
            )

    def forward(self, *mods):
        """mmdrop wrapper forward

        Perform hard or soft mmdrop

        Args:
            mods (varargs torch.Tensor): [B, L, D_m] Modality representations

        Returns:
            (List[torch.Tensor]): The modality representations. Some of them are dropped

        """
        return self.mmdrop(*mods)
