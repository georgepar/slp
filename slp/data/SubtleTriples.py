from torch.utils.data import Dataset


class SubTriples1(Dataset):
    def __init__(self, directory, transforms=None, train=True):

        self.ids, self.triples = self.read_data(directory)
        self.transforms = transforms

    def read_data(self, directory):
        lines = open(directory).read().split("\n")[:-1][:10000]
        triplets = []
        ids = []
        for index, line in enumerate(lines):
            if not line == '':
                if line[0] == 'S':
                    ids.append(line.split()[2])
                elif line[0] == 'I':
                    u1 = line[4:]
                elif line[0] == 'R':
                    u2 = line[4:]

            if index % 6 == 0 and line == '' and (not index == 0):
                triplets.append((u1, u2, ''))

        return ids, triplets

    def map(self, t):
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        id = self.ids[idx]
        #print(id)
        s1, s2, s3 = self.triples[idx]
        if self.transforms is not None:
            for t in self.transforms:
                s1 = t(s1)
                s2 = t(s2)
                s3 = t(s3)

        return s1, s2, s3


class SubTriples2(Dataset):
    def __init__(self, directory, transforms=None, train=True):

        self.ids, self.triples = self.read_data(directory)
        self.transforms = transforms

    def read_data(self, directory):
        lines = open(directory).read().split("\n")[:-1][:10000]
        triplets = []
        ids = []
        for index, line in enumerate(lines):
            if not line == '':
                if line[0] == 'S':
                    ids.append(line.split()[2])
                elif line[0] == 'I':
                    u2 = line[4:]
                elif line[0] == 'R':
                    u3 = line[4:]

            if index % 6 == 0 and line == '' and (not index == 0):
                triplets.append(('', u2, u3))

        return ids, triplets

    def map(self, t):
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        id = self.ids[idx]
        s1, s2, s3 = self.triples[idx]
        if self.transforms is not None:
            for t in self.transforms:
                s1 = t(s1)
                s2 = t(s2)
                s3 = t(s3)

        return s1, s2, s3

if __name__ == '__main__':
    dataset = SubTriples2('./data/corpus0sDialogues.txt')
    for d in dataset:
        print(d)


