import torch.nn as nn

from slp.modules.feedforward import FF


class Classifier(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.clf = FF(encoded_features, num_classes,
                      activation='none', layer_norm=False,
                      dropout=0.)

    def forward(self, x):
        x = self.encoder(x)
        return self.clf(x)
