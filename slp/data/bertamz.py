import os
import torch

from torch.utils.data import Dataset
from transformers import *
from slp.util import mktensor


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocessing (sequence, tokenizer):
    #import ipdb; ipdb.set_trace()
    sequence.insert(0, "[CLS]")
    sequence.insert(len(sequence), "[SEP]")
    text = tokenizer.convert_tokens_to_ids(sequence)
    text = mktensor(text, device=DEVICE, dtype=torch.long)
    return text

class AmazonZiser17(Dataset):
    def __init__(self, ds="books", dl=0, labeled=True):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.labels = []
        self.reviews = []
        self.domains = []
        self.transforms = []
        self.domain = dl 
        if labeled:
            labf = "all.txt"
        else:
            labf = "unl.txt"  
        file = os.path.join("../slpdata/amazon/ziser17", ds, labf)
        with open(file) as f:
            for row in f:
                if labf == "unl.txt":
                    label = -1
                    review = row[3:]
                else:
                    label, review = int(row[0]), row[2:]
                if len(review)>10000:
                   review = review[:10000]
                self.labels.append(label)
                self.reviews.append(review)
                self.domains.append(self.domain)

    def map(self, t):
        self.transforms.append(t)
        return self
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = self.reviews[idx]
        review = torch.tensor(self.tokenizer.encode(review, add_special_tokens=True), dtype=torch.long)
        label = self.labels[idx]
        domain = self.domains[idx]
        return review, label#, domain

if __name__ == '__main__':
    data = AmazonZiser17()
    i  = 0 
    maxd = 0
    for d in data:
        i = i + 1
        r, l = d
        print (len(r))
        if len(r)> maxd:
            maxd = len(r)
    print (i)
    print (maxd)
