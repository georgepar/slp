import torch.nn as nn


class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self, pad_idx=0):
        super(SequenceCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, y_pred, targets): 
        #targets = targets[:, 1:]
        # y_pred = y_pred[1:].reshape(-1, y_pred.size(-1))
        # targets = targets[1:].reshape(-1)
        y_pred = y_pred.contiguous().view(-1, y_pred.size(-1))
        targets = targets.contiguous().view(-1)
        return self.criterion(y_pred, targets)
