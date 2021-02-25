import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.clf = nn.Linear(encoded_features, num_classes)

    def forward(self, *args, **kwargs):
        x = self.encoder(*args, **kwargs)
        return self.clf(x)
