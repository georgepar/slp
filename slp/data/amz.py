import os
from torch.utils.data import Dataset


class AmazonZiser17(Dataset):
    def __init__(self, ds="books", dl=0, labeled=True):
        self.labels = []
        self.reviews = []
        self.domains = []
        self.transforms = []
        self.domain = dl 
        if labeled:
            labf = "all.txt"
        else:
            labf = "unl.txt"  
        file = os.path.join("./data/amazon/ziser17", ds, labf)
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
        label = self.labels[idx]
        domain = self.domains[idx]
        for t in self.transforms:
            review = t(review)
        return review, label, domain

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
