import pandas as pd
import os
import csv

from html.parser import HTMLParser
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset


def pad_sequence(sequences, batch_first=False, padding_len=None, padding_value=0):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
#    import pdb; pdb.set_trace()
    max_size = sequences[0].size()

    trailing_dims = max_size[1:]
    if padding_len is not None:
        max_len = padding_len
    else:
        max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        if tensor.size(0) > padding_len:
            tensor = tensor[:padding_len]
        length = min(tensor.size(0), padding_len)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor
    return out_tensor



class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


class TupleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx]
        return sample, label


class PsychologicalDataset(Dataset):
    def __init__(self, csv_file, root_dir, max_word_len=25, text_transforms=None):
        self.file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.max_word_len = max_word_len
        self.text_transforms = text_transforms
        self.transcript, self.label, self.metadata, self.title = self.get_files_labels_metadata(self.root_dir, self.file)
        self.patient_turns = ['CLIENT','PT','PATIENT','CL','Client','Danny','Juan',
                                'PARTICIPANT','CG', 'RESPONDENT','F','Angie','Jeff', 'Bill']


    def get_files_labels_metadata(self, root_dir, _file):
        included_cols = [1,11,12,13,14,15,16,20,22]
        #included_cols_names = ['file name', 'session title', 'client gender', 
        #                       'client age', 'client marital status', 
        #                       'client sexual orientation', 'therapist gender', 
        #                       'therapist experience', 'psych. subject', 'therapies']

        transcripts = []
        labels = []
        title = []
        metadata = []

        rows = []
        _file = "../data/balanced_new_csv.csv"
        with open(_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                rows.append(list(row))
        
        for i in range(len(rows)):
            labels.append(rows[i][21])
            title.append(rows[i][5])
            metadata.append([rows[i][y] for y in included_cols])

        for i in range(len(rows)):
            f = int(float(rows[i][1]))
            filename = str(os.path.join(self.root_dir, str(f))) + '.txt'
            fp = open(filename,'r+')
            transcript = fp.read()
            transcripts.append(transcript)

        return transcripts, labels, metadata, title


    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):
        preprocessed_text = self.transcript[idx]
        label = self.label[idx].split("; ")
        title = self.title[idx]
        metadata = self.metadata[idx]

        if self.text_transforms is not None:
            lista = []
            turns = []
            p = strip_tags(preprocessed_text)
            p = p.split("\n")
            p1 = [x for x in p if x!='']


            for (i, j) in zip(p1[::2], p1[1::2]):
                i = i.split(":")
                j = j.split(":")
                if len(i)!= 1 and len(j)!= 1:
                    turns.append(i[0])
                    turns.append(j[0])
                    d = i[1] + ' ' + j[1]
                    lista.append(self.text_transforms(d))

            if len(lista) == 0:
#                import pdb; pdb.set_trace()
                for (i, j) in zip(p1[::2], p1[1::2]):
                    i = i.split(":")
                    j = j.split(":")
                    if len(i)!= 1:
                        turns.append(i[0])
                        isum = i[1]
                    else:
                        isum = ''

                    if len(j)!= 1:
                        turns.append(j[0])
                        jsum = j[1]
                    else:
                        jsum = ''

                    d = isum + ' ' + jsum
                    lista.append(self.text_transforms(d))
#            if len(lista) == 0:
#                import pdb; pdb.set_trace()


#            import pdb; pdb.set_trace()
            preprocessed_text = pad_sequence(lista, batch_first=True, padding_len=self.max_word_len)
            preprocessed_title = self.text_transforms(title)

        lab = int("Depression (emotion)" in label)
        return (preprocessed_text, preprocessed_title, lab)

