import os

BASE_DIR = '../data/'


# Opinion Lexicon (or Sentiment Lexicon) -  Bing Liu (~6.800 entries)
# --------------------------------------
# format = dictionary with entries like this:
# word1={'positive': 1, 'negative': 0}
# word2={'positive': 0, 'negative': 1}

def load_bingliu_lexicon(neg_file, pos_file):

    # returns Bing Liu Opinion lexicon in the form of a dictionary
    # keys: words, values: "positive" or "negative"

    _data = {}

    # negative words
    lines = open(neg_file, "r", encoding="utf-8").readlines()
    lines = lines[35:]

    total_neg_words = len(lines)

    for line_id, line in enumerate(lines):
        _row = line.rstrip().split('\t')
        _word = _row[0]
        _feature = "negative"
        _data[_word] = _feature


    # positive words
    lines = open(pos_file, "r", encoding="utf-8").readlines()
    lines = lines[35:]

    total_pos_words = len(lines)
    cnt = 0
    for line_id, line in enumerate(lines):
        _row = line.rstrip().split('\t')
        _word = _row[0]

        if _word in _data.keys():
            cnt += 1
        _feature = "positive"
        _data[_word] = _feature

    return _data, cnt, total_neg_words, total_pos_words

####################################################
# Load Bing Liu Opinion Lexicon
####################################################

# get the Bing Liu Opinion Lexicon in the form of a dictionary
# where keys are the unique words
# and values a scalar

def bing_liu():
 #   BL_LEX_PATH = os.path.join(BASE_DIR, 'lexicons_kate', 'Bing_Liu_opinion_lex')
    BL_LEX_PATH = BASE_DIR
    lexicon, both_pos_neg, neg_words, pos_words = load_bingliu_lexicon(neg_file=os.path.join(BL_LEX_PATH, 'negative-words.txt'),
                               pos_file=os.path.join(BL_LEX_PATH, 'positive-words.txt'))
    lex = {}
    for word in lexicon:
        if lexicon[word] == 'negative':
            lex[word] = [-1.]
        elif lexicon[word] == 'positive':
            lex[word] = [1.]
    return lex

