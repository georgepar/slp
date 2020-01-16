import os
import codecs
import csv
import unicodedata
import re
import numpy as np
from torch.utils.data import Dataset
from slp.data.transforms import *


class DummyMovieTriples(Dataset):
    def __init__(self, directory, transforms=None, train=True):

        self.triples = self.read_data(directory)
        self.transforms = transforms


    def read_data(self, directory):
        lines = open(directory).read().split("\n")[:-1]
        triplets=[]
        for line in lines:
            triplets.append(line.split('/'))
        return triplets


    def map(self,t):
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        s1,s2,s3 = self.triples[idx]

        if self.transforms is not None:
            s1 = self.transforms(s1)
            s2 = self.transforms(s2)
            s3 = self.transforms(s3)

        return s1,s2,s3

if __name__=='__main__':
    dataset = DummyMovieTriples('./data/dummy_movie_triples')
    for d in dataset:
        print(d)