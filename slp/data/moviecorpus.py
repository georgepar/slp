import os

from zipfile import ZipFile

from torch.utils.data import Dataset

from slp.config.moviecorpus import MOVIECORPUS_URL
from slp.util.system import download_url
from slp.data.transforms import *


class MovieCorpusDataset(Dataset):
    def __init__(self, directory, transforms=None, train=True):
        dest = download_url(MOVIECORPUS_URL, directory)
        with ZipFile(dest, 'r') as zipfd:
            zipfd.extractall(directory)
        self._file_lines = os.path.join(directory,
                                        'cornell '
                                        'movie-dialogs '
                                        'corpus',
                                        'movie_lines.txt')

        self._file_convs = os.path.join(directory,
                                        'cornell '
                                        'movie-dialogs '
                                        'corpus',
                                        'movie_conversations.txt')

        self.transforms = transforms
        self.pairs = self.get_metadata()
        self.transforms = transforms

    def get_metadata(self):
        movie_lines = open(self._file_lines, encoding='utf-8',
                           errors='ignore').read().split('\n')
        movie_conv_lines = open(self._file_convs, encoding='utf-8',
                                errors='ignore').read().split('\n')

        # Create a dictionary to map each line's id with its text
        id2line = {}
        for line in movie_lines:
            _line = line.split(' +++$+++ ')
            if len(_line) == 5:
                id2line[_line[0]] = _line[4]

        # Create a list of all of the conversations lines ids.
        convs = []
        for line in movie_conv_lines[:-1]:
            _line = line.split(' +++$+++ ')[-1][1:-1]\
                .replace("'", "").replace(" ", "")
            convs.append(_line.split(','))

        # Sort the sentences into questions (inputs) and answers (targets)
        questions = []
        answers = []

        for conv in convs:
            for i in range(len(conv) - 1):
                questions.append(id2line[conv[i]])
                answers.append(id2line[conv[i + 1]])
        return list(zip(questions, answers))

    def filter_data(self,min_threshold_length,max_threshold_length):
        """
        This method filters data according to threshold length.
        If the length of the turn exceeds threshold length the data is deleted.
        """
        new_questions = []
        new_answers = []
        for question,answer in self.pairs:
            if len(question)<=max_threshold_length and len(question)>=min_threshold_length and len(answer)<=max_threshold_length and len(answer)>=min_threshold_length:
                new_questions.append(question)
                new_answers.append(answer)
        
        self.pairs = list(zip(new_questions,new_answers))
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        question, answer = self.pairs[idx]

        if self.transforms is not None:
            question = self.transforms(question)
            answer = self.transforms(answer)

        return question, answer


if __name__ == '__main__':
    from slp.util.embeddings import EmbeddingsLoader
    from torchvision.transforms import Compose
    loader = EmbeddingsLoader(
        './cache/glove.6B.50d.txt', 50)
    word2idx, idx2word, embeddings = loader.load()

    transforms = Compose([SpacyTokenizer(), ToTokenIds(word2idx)])
    data = MovieCorpusDataset('./data/', transforms=transforms)
    
    print(len(data))
    print(data[5])
    data.filter_data(3,10)
    print(len(data))
    print(data[5])
    
