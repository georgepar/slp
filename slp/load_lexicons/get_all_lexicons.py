import os
import torch

import torch.nn as nn

from get_afinn_features import load_afinn_lexicon
from get_BL_features import *
from get_liwc_features import load_liwc_lex , load_features
from get_mpqa_features import *
from get_semeval2015_twitter_features import *
from get_nrc_emolex_features import *
from slp.data.therapy_title import pad_sequence
from slp.data.transforms import ToTensor

BASE_DIR = '../data/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#def LexiconFeatures(inputs):

class LexiconFeatures(nn.Module):
    def __init__(self):
        super(LexiconFeatures, self).__init__()

        self.afinn = load_afinn_lexicon()
        self.BL = bing_liu()
        self.liwc = load_liwc_lex()
        self.liwc_classes, _ = load_features(os.path.join(BASE_DIR, 'PsycholinguisticDimensions.txt'))
        self.mpqa = mpqa_lex()
        self.semeval = semeval15_lexicon()
        self.emolex = emolex()

    def forward(self, inputs, idx2word, padding_len):
        to_tensor = ToTensor(device=DEVICE)
        final_vector = []
#        import pdb; pdb.set_trace()
        for inputt in inputs:
            vec = []
            for inp in inputt:
                word = idx2word[inp]
                vector = []
                inp = inp.item()
                if word in self.afinn:
                    vector.append(float(self.afinn[word]))
                else:
                    vector.append(float(0))
                if word in self.semeval:
                    vector.append(self.semeval[word])
                else:
                    vector.append(float(0))
                if word in self.BL:
#                    import pdb; pdb.set_trace()
                    vector.append(float(self.BL[word[0]))
                else:
                    vector.append(float(0))
                if word in self.mpqa:
                    vector.extend(self.mpqa[word])
                else:
                    vector.extend([float(0)]*4)
                if word in self.liwc:
                    v = [float(i) for i in self.liwc[word]]
                    vector.extend(v)
                else:
                    vector.extend([float(0)]*73)
                if word in self.emolex:
                    v = [float(i) for i in self.emolex[word]]
                    vector.extend(v)
                else:
                    vector.extend([float(0)]*19)
                vec.append(vector)
            try:
                final_vector.append(to_tensor(vec))
            except:
                import pdb; pdb.set_trace()
        final_vector = pad_sequence(final_vector, padding_len=padding_len, batch_first=True)
        return final_vector
