import argparse

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import constants
from slp.config import SPECIAL_TOKENS
from slp.data.collators import Sequence2SequenceCollator
from slp.data.spelling import SpellCorrectorPredictionDataset
# from slp.data.transforms import CharacterTokenizer
from slp.data.transforms import CharacterTokenizer, WordpieceTokenizer
from slp.modules.convs2s import Seq2Seq


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


config = {
    "device": "cpu",
    "parallel": True,
    "num_workers": 2,
    "batch_size": 128,
    "lr": 1e-3,
    "max_steps": 200000,
    "schedule_steps": 1000,
    "checkpoint_steps": 10000,
    "hidden_size": 512,
    "embedding_size": 256,
    "encoder_kernel_size": 3,
    "decoder_kernel_size": 3,
    "encoder_layers": 10,
    "decoder_layers": 10,
    "encoder_dropout": 0.3,
    "decoder_dropout": 0.3,
    "max_length": 100,
    "gradient_clip": 0.1,
    # "teacher_forcing": 0.4,
}



def parse_args():
    parser = argparse.ArgumentParser("Evaluate spell checker")
    parser.add_argument(
        "--sentences",
        type=str,
        help="Test file containing one or more misspelled sentences",
    )
    parser.add_argument("--ckpt", type=str, help="Checkpoint")
    args = parser.parse_args()

    return args


collate_fn = Sequence2SequenceCollator(device="cpu")


def make_corrector(model, tokenizer, max_length=256):
    def predict(sentence, sos_idx=1):
        """Predictor closure
        Target is provided by dataloader and contains only <sos><eos> tokens
        This can be created in here for speedier implementation
        """
        with torch.no_grad():
            encoder_conved, encoder_combined = model.module.encoder(sentence)

        target_indexes = [sos_idx]

        for i in range(max_length):
            target = torch.LongTensor(target_indexes).unsqueeze(0)
            with torch.no_grad():
                decoded, _ = model.module.decoder(
                    target, encoder_conved, encoder_combined
                )
            predicted_idx = decoded.argmax(-1)[0, -1].item()
            target_indexes.append(predicted_idx)
            if predicted_idx == 2:
                break
        # Greedy decoding here.
        predicted = tokenizer.detokenize(target_indexes)

        return predicted

    return predict


if __name__ == "__main__":
    args = parse_args()

    tokenizer = WordpieceTokenizer(
        lower=True,
        bert_model="nlpaueb/bert-base-greek-uncased-v1",
        prepend_bos=True,
        append_eos=True,
        specials=SPECIAL_TOKENS,
    )

    sos_idx = 1  # tokenizer.c2i[SPECIAL_TOKENS.BOS.value]
    pad_idx = 0  # tokenizer.c2i[SPECIAL_TOKENS.PAD.value]
    eos_idx = 2  # tokenizer.c2i[SPECIAL_TOKENS.EOS.value]


    # tokenizer = CharacterTokenizer(
    #     constants.CHARACTER_VOCAB,
    #     prepend_bos=True,
    #     append_eos=True,
    #     specials=SPECIAL_TOKENS,
    # )

    # sos_idx = tokenizer.c2i[SPECIAL_TOKENS.BOS.value]
    # pad_idx = tokenizer.c2i[SPECIAL_TOKENS.PAD.value]
    # eos_idx = tokenizer.c2i[SPECIAL_TOKENS.EOS.value]

    vocab_size = len(tokenizer.vocab)
    # print(tokenizer.vocab)
    testset = SpellCorrectorPredictionDataset(args.sentences, tokenizer=tokenizer)

    model = Seq2Seq(
        vocab_size,
        vocab_size,
        hidden_size=config["hidden_size"],
        embedding_size=config["embedding_size"],
        encoder_layers=config["encoder_layers"],
        decoder_layers=config["decoder_layers"],
        encoder_kernel_size=config["encoder_kernel_size"],
        decoder_kernel_size=config["decoder_kernel_size"],
        max_length=config["max_length"],
        device=config["device"],
        pad_idx=pad_idx,
        # teacher_forcing_p=config["teacher_forcing"],
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model = WrappedModel(model)
    state_dict = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(config["device"])
    model.eval()

    spell_corrector = make_corrector(model, tokenizer, max_length=config["max_length"])

    for sent, tgt in testset:
        sent, tgt = sent.unsqueeze(0), tgt.unsqueeze(0)  # Mock batch dimension
        prediction = spell_corrector(sent, sos_idx=sos_idx)
        print("Input:")
        print(tokenizer.detokenize(sent.squeeze(0).tolist()))
        print("Output:")
        print(prediction)
        print()
