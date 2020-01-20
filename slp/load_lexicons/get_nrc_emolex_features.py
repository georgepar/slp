import os
import pickle

# NRC Emotion Lexicon (Emolex)
# Total words: 14,182
# dictionary: {word: {'fear':_, 'joy':_, 'positive':_, 'emotions':(list of len 8), 'sadness':_,
# 'negative':_, 'anticipation':_, 'polarity':_, 'anger':_, 'disgust':_, 'trust':_, 'surprise':_}}

BASE_DIR = '../data/'

def emolex():
    path = os.path.join(BASE_DIR, 'emolex.pickle')
    with open(path, 'rb') as f:
        data = pickle.load(f)

    lex = {}
    for word in data:
        features = []
        for key in data[word]:
            if not isinstance(data[word][key], list):
                features.append(data[word][key])
            else:
                features += data[word][key]
        lex[word]=features


    return lex
