import torch
import torch.nn as nn


class LayerNormTf(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        Link: https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L234
        """
        super(LayerNormTf, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate Layernorm the tf way"""
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


# Default to pytorch layernorm
LayerNorm = nn.LayerNorm
