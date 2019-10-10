import torch
import torch.nn as nn

from ignite.metrics import Loss, Accuracy
from sklearn.preprocessing import LabelEncoder

from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from torchnlp.datasets import smt_dataset  # type: ignore

from slp.data.collators import SequenceClassificationCollator
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor
from slp.trainer import SequentialTrainer
from slp.util.embeddings import EmbeddingsLoader


class LSTMClassifier(nn.Module):
    def __init__(self, embeddings, hidden_dim, output_size):
        super(LSTMClassifier, self).__init__()

        self.vocab_size = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.weight = nn.Parameter(torch.as_tensor(embeddings),
                                             requires_grad=False)

        self.lstm = nn.LSTM(self.embedding_dim,
                            hidden_dim,
                            num_layers=1,
                            batch_first=True)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.dropout_layer = nn.Dropout(p=0.2)

    def init_hidden(self, batch_size):
        return(torch.autograd.Variable(
                    torch.randn(1, batch_size, self.hidden_dim)),
               torch.autograd.Variable(
                   torch.randn(1, batch_size, self.hidden_dim)))

    def forward(self, batch, lengths):
        self.hidden = self.init_hidden(batch.size(0))

        embeds = self.embedding(batch)
        packed_input = pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False)
        outputs, (ht, ct) = self.lstm(packed_input, self.hidden)

        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        return output


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
    word2idx, idx2word, embeddings = loader.load()

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

    model = LSTMClassifier(embeddings, 256, 3)
    optimizer = SGD(model.parameters(), lr=1e-3)
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
