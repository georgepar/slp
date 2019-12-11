import numpy as np
import torch
import collections
import os

from ignite.metrics import Loss
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import Compose

from slp.util.embeddings import EmbeddingsLoader
from slp.data.moviecorpus import MovieCorpusDataset
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor
from slp.data.collators import Seq2SeqCollator
from slp.trainer.trainer import Seq2SeqTrainer
from slp.config.moviecorpus import SPECIAL_TOKENS
from slp.modules.loss import SequenceCrossEntropyLoss
from slp.modules.seq2seq import EncoderDecoder, EncoderLSTM, DecoderLSTMv2

from torch.optim import Adam

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COLLATE_FN = Seq2SeqCollator(device='cpu')
MAX_EPOCHS = 10
BATCH_TRAIN_SIZE = 32
BATCH_VAL_SIZE = 32
min_threshold = 3
max_threshold = 16
max_target_len = max_threshold


def create_vocabulary_dict(dataset, tokenizer=SpacyTokenizer()):
    """
    receives dataset and a tokenizer in order to split sentences and create
    a dict-vocabulary with words counts.
    """
    voc_counts = {}
    for question, answer in dataset.pairs:
        words, counts = np.unique(np.array(tokenizer(question)),
                                  return_counts=True)
        for word, count in zip(words, counts):
            if word not in voc_counts.keys():
                voc_counts[word] = count
            else:
                voc_counts[word] += count

    return voc_counts


def create_emb_file(new_emb_file, old_emb_file, freq_words_file, mydataset,
                    tok=SpacyTokenizer(), most_freq=None):

    voc = create_vocabulary_dict(mydataset, tok)

    sorted_voc = sorted(voc.items(), key=lambda kv: kv[1])
    if not os.path.exists(freq_words_file):
        with open(freq_words_file, "w") as file:
            if most_freq is not None:
                for item in sorted_voc[-most_freq:]:
                    file.write(item[0]+'\n')
            else:
                for item in sorted_voc:
                    file.write(item[0]+'\n')
        file.close()

        os.system("awk 'FNR==NR{a[$1];next} ($1 in a)' " + freq_words_file +
                  " " + old_emb_file + ">" + new_emb_file)


def dataloaders_from_indices(dataset, train_indices, val_indices, batch_train,
                             batch_val):
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_train,
        sampler=train_sampler,
        drop_last=False,
        collate_fn=COLLATE_FN)
    val_loader = DataLoader(
        dataset,
        batch_size=batch_val,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=COLLATE_FN)
    return train_loader, val_loader


def train_test_split(dataset, batch_train, batch_val,
                     test_size=0.2, shuffle=True, seed=None):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_size * dataset_size))
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices = indices[test_split:]
    val_indices = indices[:test_split]
    return dataloaders_from_indices(dataset, train_indices, val_indices,
                                    batch_train, batch_val)


def trainer_factory(embeddings, pad_index, bos_index, device=DEVICE):
    encoder = EncoderLSTM(embeddings, emb_train=False, hidden_size=512,
                          num_layers=2, bidirectional=True, dropout=0.4,
                          rnn_type='lstm', device=DEVICE)

    decoder = DecoderLSTMv2(weights_matrix=embeddings, emb_train=False,
                            hidden_size=512,
                            output_size=embeddings.shape[0],
                            max_target_len=max_target_len, num_layers=1,
                            dropout=0.4, rnn_type='lstm',
                            emb_layer=None, bidirectional=False,
                            device=DEVICE)
    model = EncoderDecoder(
        encoder, decoder, bos_index, teacher_forcing_ratio=0.9, device=DEVICE)

    # import torch.nn as nn
    # embedding = nn.Embedding(embeddings.shape[0], 500)
    # encoder = EncoderRNN(500, embedding, 1, 0.5)
    # decoder = LuongAttnDecoderRNN('dot', embedding, 500,
    #                               embeddings.shape[0], 1, 0.5)
    #
    # model = Seq2Seq(encoder,decoder,device=DEVICE)

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-6)

    criterion = SequenceCrossEntropyLoss(pad_index)

    metrics = {
        'loss': Loss(criterion)
    }

    trainer = Seq2SeqTrainer(model,
                             optimizer,
                             checkpoint_dir=None,  # '../checkpoints',
                             metrics=metrics,
                             non_blocking=True,
                             retain_graph=False,
                             patience=5,
                             device=device,
                             clip=1.,
                             loss_fn=criterion)
    return trainer


def input_interaction(model, transforms, idx2word, pad_index, bos_index,
                      eos_index):

    model.eval()
    input_sentence = ""
    while True:

    # Get input response:
        input_sentence = input(">")
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit':
            break

        input_seq = transforms(input_sentence)
        model_outputs = model.evaluate(input_seq,eos_index)
        answer = ""
        for output in model_outputs:

            answer += idx2word[output.item()]+" "
        print(answer)

if __name__ == '__main__':

    new_emb_file = './cache/new_embs.txt'
    old_emb_file = './cache/glove.6B.300d.txt'
    freq_words_file = './cache/freq_words.txt'

    dataset = MovieCorpusDataset('./data/', transforms=None)
    dataset.filter_data(min_threshold, max_threshold)
    create_emb_file(new_emb_file, old_emb_file, freq_words_file, dataset,
                    SpacyTokenizer())

    loader = EmbeddingsLoader(new_emb_file, 300, extra_tokens=SPECIAL_TOKENS)
    word2idx, idx2word, embeddings = loader.load()

    pad_index = word2idx[SPECIAL_TOKENS.PAD.value]
    bos_index = word2idx[SPECIAL_TOKENS.BOS.value]
    eos_index = word2idx[SPECIAL_TOKENS.EOS.value]

    tokenizer = SpacyTokenizer(prepend_bos=True,
                               append_eos=True,
                               specials=SPECIAL_TOKENS)
    to_token_ids = ToTokenIds(word2idx)
    to_tensor = ToTensor(device='cpu')

    transforms = Compose([tokenizer, to_token_ids, to_tensor])
    dataset = MovieCorpusDataset('./data/', transforms=transforms)
    dataset.filter_data(min_threshold, max_threshold)

    train_loader, val_loader = train_test_split(dataset, BATCH_TRAIN_SIZE,
                                                BATCH_VAL_SIZE)
    trainer = trainer_factory(embeddings, pad_index, bos_index, device=DEVICE)
    final_score = trainer.fit(train_loader, val_loader, epochs=MAX_EPOCHS)
    print(f'Final score: {final_score}')

    # input_interaction(trainer.model, transforms, idx2word, pad_index, bos_index,
    #                   eos_index)
    #trainer.overfit_single_batch(train_loader)
