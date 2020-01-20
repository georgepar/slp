import os

#from sys_config import BASE_DIR
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = '../data/'
# LIWC Lexicon http://lit.eecs.umich.edu/~geoliwc/LIWC_Dictionary.htm

def load_liwc_lexicon(file):
    # returns LIWC in the form of a dictionary
    # keys: words, values: feature vector (list)


    _data = {}

    lines = open(file, "r", encoding="utf-8").readlines()
    for line_id, line in enumerate(lines):
        _row = line.rstrip().split(" ")
        _word = _row[0]
        _features = _row[1:]
        _data[_word] = _features
    return _data


def load_features(file):

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
# Load LIWC Lexicon
####################################################

def liwc_lex():
    # get the liwc lexicon in the form of a dictionary
    # where keys are the unique words
    # and values a list with all the dimensions (73 in total)


    lex = load_liwc_lexicon(
        os.path.join(BASE_DIR, 'PsycholinguisticLexicon.txt'))

    total_words = len(lex)

    # get the two dictionaries that relate every dimension name
    # with its corresponding number (value) in the lexicon dimension list
    dim2num, num2dim = load_features(
        os.path.join(BASE_DIR, 'PsycholinguisticDimensions.txt'))

    ####################################################
    # Plot statistics of LIWC Lexicon
    ####################################################

    # The lexiconss has 18504 words and for each word a feature vector of size 71.
    # Each dimension represents a category (for example affect, posemo, negemo etc)
    # The vector contains '1' when this word is includied in the particular category.
    # Otherwise '0'.
    # Using a bar plot we can decide which dimensions of this feature vector are useful for our work.

    # initialization of count dictionary
    dimensions = list(dim2num.keys())
    dim_counts = {dim: 0 for dim in dimensions}

    for word in lex:
        ones = [i for i, x in enumerate(lex[word]) if x == '1']
        for index in ones:
            dim_counts[num2dim[index]] += 1

    sorted_tuples = sorted(dim_counts.items(), key=lambda kv: kv[1])

    x = [k[1] for k in sorted_tuples if k[1] > 500]
    y = [k[0] for k in sorted_tuples if k[1] > 500]


    plt.figure()
    sns.barplot(x=x, y=y)
    plt.title('Number of words for each dimension of the LIWC lexicon')
    # plt.show()
    plt.savefig('liwc_dims_statistics.png')
    # plt.close()

    print(len(lex))


def load_liwc_lex():
    return load_liwc_lexicon(
        os.path.join(BASE_DIR, 'PsycholinguisticLexicon.txt'))

 #   liwc_lex()
