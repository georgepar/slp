import os
import pickle

BASE_DIR = '../data/'

# SemEval-2015 English Twitter Sentiment Lexicon
# aka NRC MaxDiff Twitter Sentiment Lexicon
# Total words: 1515 (including hashtags like #ew)
# dictionary: {word: real value -1 to +1, representing negative/positive sentiment}

def semeval15_lexicon():
    path = os.path.join(BASE_DIR, 'SemEval2015-English-Twitter-Lexicon.pickle')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


# print(len(data))
