from torch.utils.data import Dataset


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

    def map(self, t):
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        s1, s2, s3 = self.triples[idx]

        if self.transforms is not None:
            for t in self.transforms:
                s1 = t(s1)
                s2 = t(s2)
                s3 = t(s3)

        return s1, s2, s3


if __name__ == '__main__':
    dataset = DummyMovieTriples('./data/dummy_movie_triples')
    for d in dataset:
        print(d)
