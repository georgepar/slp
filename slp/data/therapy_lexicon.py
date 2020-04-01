import pandas as pd
import os
import csv

from html.parser import HTMLParser
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from itertools import groupby

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
        self._file = csv_file
        self.root_dir = root_dir
        self.max_word_len = max_word_len
        self.transforms = text_transforms
        self.transcript, self.label, self.metadata, self.title = self.get_files_labels_metadata(self.root_dir, self._file)
        self.patient_turns = ['FEMALE CLIENT', 'MALE CLIENT', 'Audience', 'CLIENT','PT','PATIENT','CL','Client','Danny','Juan', 'MAN',
                                'PARTICIPANT', 'CG', 'MAN', 'RESPONDENT','F','Angie','Jeff', 'Bill', 'Jim', 'Leah', 'Kelly', 'MRS. NAVNOR', 
				'MR. NAVNOR', 'MICHELLE', 'Phil', 'FEMALE PARTICIPANT', 'Mom', 'Nicole', 'LINDA', 
				'MALE PARTICIPANT', 'Blake', 'M', 'Claudette', 'MR. VAC', 'Marie', 'Robin', 'Mike', 'Gina', 'FEMALE', 'LORI'
                                ,'Joshua', 'Shayla', 'Greg', 'Barbara', 'MARGE', 'ANN LARKIN', 'EDWARD', 'Mark', 'PATiENT']
        self.therapist_turns = ['CONNIRAE', 'ANALYST', 'THERAPIST','COUNSELOR','DR','M','Therapist','Marlatt', 'Lazarus','INTERVIEWER',
				'TH', 'Johnson', 'Scharff', 'T', 'Counselor', 'Wubbolding', 'DR. WARKENTIN', 'Bugental', 'Powers', 'Koocher',
				'Dr. Sklare', 'BECKER', 'Hardy', 'MODERATOR', 'Masek', 'VIRGINIA','MODERATOR', 'Oaklander', 'McCrady', 
                                'Bugental', 'Krumboltz', 'Miller', 'ANDREAS', 'Kottman', 'Utigaard', 'Wubbolding', 'Carlson', 'JOSH LOMAN',
                                'Zweben']


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
#        _file = "../data/balanced_new_csv.csv"
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

        mean_length = 0
        import re
        if self.transforms is not None:
            lista = []
            turns = []
            total_turns = []
            p = strip_tags(preprocessed_text)
            p = p.split("\n")
            p1 = [x for x in p if x!='']
#            import pdb; pdb.set_trace()
            for i in p1:
                i = i.split(":")
                if len(i) >= 2:
                    lista.extend(self.transforms(i[1]))

            if len(lista) == 0:
                import pdb; pdb.set_trace()

            preprocessed_text = lista
            preprocessed_title = self.transforms(title)

        lab = int("Depressive disorder" in metadata[7] or "Depressive disorder" in label 
		or "Depression (emotion)" in label or "Depression (eotion)" in metadata[7])

        return (preprocessed_text, preprocessed_title, lab)

