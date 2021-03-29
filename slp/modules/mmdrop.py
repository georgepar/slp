import random
from typing import List, Optional

import torch
import torch.nn as nn


class HardMultimodalDropout(nn.Module):
    def __init__(
        self,
        p: float = 0.5,
        n_modalities: int = 3,
        p_mod: Optional[List[float]] = None,
        masking: bool = False,
        m3_sequential: bool = False,
    ):
        """MMDrop initial implementation

        For each sample in a batch drop one of the modalities with probability p

        Args:
            p (float): drop probability
            n_modalities (int): number of modalities
            p_mod (Optional[List[float]]): Drop probabilities for each modality
            masking (bool): masking flag variable
            m3_sequential (bool): mask different instances of the sequence for each modality
        """
        super(HardMultimodalDropout, self).__init__()
        self.p = p
        self.n_modalities = n_modalities
        self.masking = masking
        self.m3_sequential = m3_sequential

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

                if self.m3_sequential:
                    #  naive implementation but it works
                    # uncomment in case of disaster
                    # for batch in range(mods[0].size(0)):
                    #     for timestep in range(mods[0].size(1)):
                    #         m = random.choices(
                    #             list(range(self.n_modalities)), weights=self.p_mod, k=1
                    #         )[0]
                    #     mods[m][batch, timestep, :] = 0.0
                    bsz, seqlen = mods[0].size(0), mods[0].size(1)
                    p_modal = torch.distributions.categorical.Categorical(
                        torch.tensor(self.p_mod)
                    )
                    m_cat = p_modal.sample((bsz, seqlen)).to(mods[0].device)
                    for m in range(self.n_modalities):
                        mask = torch.where(m_cat == m, 0, 1).unsqueeze(2)
                        mods[m] = mods[m] * mask

                else:
                    for batch in range(mods[0].size(0)):
                        m = random.choices(
                            list(range(self.n_modalities)), weights=self.p_mod, k=1
                        )[0]

                        # m = random.randint(0, self.n_modalities - 1)
                        mask = torch.ones_like(mods[m])
                        mask[batch] = 0.0
                        mods[m] = mods[m] * mask

        if not self.masking:
            if self.p > 0:
                for m in range(len(mods)):
                    keep_prob = 1 - (self.p / self.n_modalities)
                    mods[m] = mods[m] * (1 / keep_prob)

        return mods


class SoftMultimodalDropout(nn.Module):
    def __init__(
        self,
        p: float = 0.5,
        n_modalities: int = 3,
        p_mod: Optional[List[float]] = None,
        masking: bool = False,
        m3_sequential: bool = False,
    ):
        """Soft mmdrop implementation

        Drop p * 100 % of features of a specific modality over batch

        Args:
            p (float): drop probability
            n_modalities (int): number of modalities
            p_mod (Optional[List[float]]): Drop probabilities for each modality
            masking: masking flag variable
            m3_sequential: use per timestep masking
        """
        super(SoftMultimodalDropout, self).__init__()
        self.p = p  # p_drop
        self.n_modalities = n_modalities
        self.masking = masking
        self.m3_sequential = m3_sequential

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
            if self.m3_sequential:
                for timestep in range(mods[0].size(1)):
                    m = random.choices(
                        list(range(self.n_modalities)), weights=self.p_mod, k=1
                    )[0]
                    binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
                    mods[m][timestep] = mods[m][timstep] * binomial.sample(
                        mods[m][timestep].size()
                    ).to(mods[m].device)
                    import pdb

                    pdb.set_trace()
            else:
                # m = random.randint(0, self.n_modalities - 1)
                m = random.choices(
                    list(range(self.n_modalities)), weights=self.p_mod, k=1
                )[0]

                binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
                mods[m] = mods[m] * binomial.sample(mods[m].size()).to(mods[m].device)

        if not self.masking:
            for m in range(self.n_modalities):
                mods[m] = mods[m] * (1.0 / (1 - self.p / self.n_modalities))

        return mods


class HardMultimodalMasking(nn.Module):
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
        super(HardMultimodalMasking, self).__init__()
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

            # if self.p > 0:
            #     for m in range(len(mods)):
            #         keep_prob = 1 - (self.p / self.n_modalities)
            #         mods[m] = mods[m] * (1 / keep_prob)

        return mods


class SoftMultimodalMasking(nn.Module):
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
        super(SoftMultimodalMasking, self).__init__()
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

            # for m in range(self.n_modalities):
            #     mods[m] = mods[m] * (1.0 / (1 - self.p / self.n_modalities))

        return mods


class MultimodalDropout(nn.Module):
    def __init__(
        self,
        p: float = 0.5,
        n_modalities: int = 3,
        p_mod: Optional[List[float]] = None,
        mode: str = "hard",
        masking: bool = False,
        m3_sequential: bool = False,
    ):
        """mmdrop wrapper class

        Drop p * 100 % of features of a specific modality over batch

        Args:
            p (float): drop probability
            n_modalities (int): number of modalities
            p_mod (Optional[List[float]]): Drop probabilities for each modality
            mode (str): Hard or soft mmdrop
            masking (bool): use m3 (no scaling)
            m3_sequential (bool): per timestep modality masking
        """
        super(MultimodalDropout, self).__init__()

        assert mode in [
            "hard",
            "soft",
        ], "Allowed mode for MultimodalDropout ['hard' | 'soft']"

        if mode == "hard":
            self.mmdrop = HardMultimodalDropout(
                p=p,
                n_modalities=n_modalities,
                p_mod=p_mod,
                masking=masking,
                m3_sequential=m3_sequential,
            )
        else:
            self.mmdrop = SoftMultimodalDropout(  # type: ignore
                p=p,
                n_modalities=n_modalities,
                p_mod=p_mod,
                masking=masking,
                m3_sequential=m3_sequential,
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


class MultimodalMasking(nn.Module):
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
        super(MultimodalMasking, self).__init__()

        assert mode in [
            "hard",
            "soft",
        ], "Allowed mode for MultimodalDropout ['hard' | 'soft']"

        if mode == "hard":
            self.mmdrop = HardMultimodalMasking(
                p=p, n_modalities=n_modalities, p_mod=p_mod
            )
        else:
            self.mmdrop = SoftMultimodalMasking(  # type: ignore
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
