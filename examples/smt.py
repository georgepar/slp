import torch
import torch.nn as nn

from ignite.metrics import Loss, Accuracy
from sklearn.preprocessing import LabelEncoder

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from torchnlp.datasets import smt_dataset  # type: ignore

from slp.data.collators import SequenceClassificationCollator
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor
from slp.modules.classifier import Classifier
from slp.modules.rnn import WordRNN
from slp.trainer import SequentialTrainer
from slp.util.embeddings import EmbeddingsLoader


class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transforms = []
        self.label_encoder = (LabelEncoder()
                              .fit([d['label'] for d in dataset]))

    def map(self, t):
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datum = self.dataset[idx]
        text, target = datum['text'], datum['label']
        target = self.label_encoder.transform([target])[0]
        for t in self.transforms:
            text = t(text)
        return text, target


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

collate_fn = SequenceClassificationCollator(device='cpu')


if __name__ == '__main__':
    loader = EmbeddingsLoader(
        '../cache/glove.840B.300d.txt', 300)
    word2idx, _, embeddings = loader.load()

    tokenizer = SpacyTokenizer()
    to_token_ids = ToTokenIds(word2idx)
    to_tensor = ToTensor(device='cpu')

    def create_dataloader(d):
        d = (DatasetWrapper(d).map(tokenizer).map(to_token_ids).map(to_tensor))
        return DataLoader(
            d, batch_size=128,
            num_workers=1,
            pin_memory=True,
            collate_fn=collate_fn)

    train_loader, dev_loader, test_loader = map(
        create_dataloader,
        smt_dataset(directory='../data/', train=True, dev=True, test=True))

    model = Classifier(
        WordRNN(256, embeddings, bidirectional=True,
                packed_sequence=True, attention=True, device=DEVICE),
        512, 3)

    optimizer = Adam([p for p in model.parameters() if p.requires_grad],
                     lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    metrics = {
        'accuracy': Accuracy(),
        'loss': Loss(criterion)
    }
    trainer = SequentialTrainer(model, optimizer,
                                checkpoint_dir='/tmp/ckpt',
                                metrics=metrics,
                                non_blocking=True,
                                patience=1,
                                loss_fn=criterion)
    trainer.fit(train_loader, dev_loader, epochs=10)
