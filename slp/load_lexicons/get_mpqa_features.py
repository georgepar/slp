import os
import pickle

BASE_DIR = '../data/'

def mpqa_lex():
    path = os.path.join(BASE_DIR, 'mpqa.pickle')
    with open(path, 'rb') as f:
        data = pickle.load(f)

    pos = list(data["reinforcement"].keys())[0]

    # dictionary in the following form:
    # {'word':
    #         {'POS':
    #                {'strength':weaksubj or strongsubj,
    #                 'positive': 0 or 1,
    #                 'negative': 0 or 1,
    #                 'polarity': 0 or 1}}}

    polarities = []
    strengths = []
    pos_tags = []
    negatives = []
    positives = []
    lexicon = {}
    feat_lexicon = {}
    for key in data:
        pos = list(data[key].keys())[0]
        lexicon[key] = {'pos': pos,
                    'strength': data[key][pos]['strength'],
                    'positive': data[key][pos]['positive'],
                    'negative': data[key][pos]['negative'],
                    'polarity': data[key][pos]['polarity']}
        polarities.append(data[key][pos]['polarity'])
        pos_tags.append(pos)
        negatives.append(data[key][pos]['negative'])
        positives.append(data[key][pos]['positive'])
        strengths.append(data[key][pos]['strength'])

        # first we add to the feature vector the subjectivity
        if data[key][pos]['strength'] == "strongsubj":
            feat_lexicon[key] = [1.0]
        elif data[key][pos]['strength'] == "weaksubj":
            feat_lexicon[key] = [0.0]
        #  then, the polarity
        feat_lexicon[key].append(float(data[key][pos]['polarity']))
        #  then, the positivity
        feat_lexicon[key].append(float(data[key][pos]['positive']))
        # and finally the negativity
        feat_lexicon[key].append(float(data[key][pos]['positive']))
    # print(len(lexicon))

    # now it is in the form: { word:{'pos':_, 'positive':_, 'negative':_, 'polarity':_} }
    # polarity: -2 to +2
    # pos: 'NOUN', 'ADJ', 'ADV', 'VERB', '_'
    # strength: weaksubj or strongsubj
    # positive/negative: 0 or 1

    # the lists are for statistics. total words: 6886

    return feat_lexicon

