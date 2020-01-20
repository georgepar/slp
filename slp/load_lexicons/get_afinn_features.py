# AFINN is a list of English words rated for valence with an integer
# between minus five (negative) and plus five (positive). The words have
# been manually labeled by Finn Årup Nielsen in 2009-2011. The file
# is tab-separated. Total words: 2477.

import os

BASE_DIR = '../data/'


def load_afinn_lexicon():

    # returns AFINN lexicon in the form of a dictionary
    # keys: words, values: valence score (integer -5 to +5)

#    file = os.path.join(BASE_DIR, 'lexicons_kate', 'AFINN', 'AFINN-111.txt')
    file = os.path.join(BASE_DIR, 'AFINN-111.txt')

    _data = {}

    lines = open(file, "r", encoding="utf-8").readlines()
    for line_id, line in enumerate(lines):
        _row = line.rstrip().split('\t')
        _word = _row[0]
        _feature = _row[1]
        _data[_word] = _feature
    return _data

def load_features(file):

    print("edw")
    dim2num = {}  # [dimension name]: corresponding number in lexicon list
    num2dim = {}  # the exact opposite

    lines = open(file, "r", encoding="utf-8").readlines()
    for line_id, line in enumerate(lines):
        _row = line.rstrip().split(" ")
        _dim = _row[1]
        dim2num[_dim] = line_id
        num2dim[line_id] = _dim
    return dim2num, num2dim

####################################################
# Load AFINN Lexicon
####################################################

# get the AFINN lexicon in the form of a dictionary
# where keys are the unique words
# and values a scalar
#
# total_words = len(lex)




