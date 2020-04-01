import torch
import torch.nn as nn
import torch.nn.functional as F

from slp.modules.helpers import PackSequence, PadPackedSequence
from slp.data.therapy import pad_sequence


#DEVICE = 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class WordAttNet(nn.Module):
    def __init__(self, dict_size, diction, hidden_size=300):
        super(WordAttNet, self).__init__()

        self.gru = nn.GRU(300, 300, bidirectional = True, batch_first=True)

        self.word = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.context = nn.Linear(2 * hidden_size, 1, bias=False)
       
        self.diction = diction
        self.dict_size = dict_size
        self.lookup = nn.Embedding(num_embeddings = self.dict_size, embedding_dim =300).from_pretrained(self.diction)
        self.pack = PackSequence(batch_first=True)
        self.unpack = PadPackedSequence(batch_first=True)

    def forward(self, inputs, lengths, hidden_state, is_title=False):
        output_emb = self.lookup(inputs)

#        import pdb; pdb.set_trace()
        
        output, lengths = self.pack(output_emb ,lengths)
        f_output, h_output = self.gru(output.float(), hidden_state)
        f_output = self.unpack(f_output, lengths)

        output = self.word(f_output)
        output = self.context(output)
        output = F.softmax(output, dim=1)
        output = (f_output * output).sum(1)
        return output, h_output


class SentAttNet(nn.Module):
    def __init__(self, hidden_size=300, num_classes=0):
        super(SentAttNet, self).__init__()
        num_classes = num_classes
        self.gru = nn.GRU(2 * hidden_size, 300, bidirectional=True, batch_first=True) #changed hidden size


        self.sent = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.context = nn.Linear(2 * hidden_size, 1, bias=False)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

        self.pack = PackSequence(batch_first=True)
        self.unpack = PadPackedSequence(batch_first=True)


    def forward(self, inputs, features, lengths, titles, hidden_state):
#        import pdb; pdb.set_trace()

        f_output, lengths = self.pack(inputs, lengths)
        f_output, h_output = self.gru(f_output, hidden_state)
        f_output = self.unpack(f_output, lengths)

        titles = torch.unsqueeze(titles, dim=1)
        f_output = torch.cat((f_output,titles), dim=1)


        output = self.sent(f_output)
        output = self.context(output)

        #features = torch.unsqueeze(features, dim=2)

        output = F.softmax(output, dim=1)
        output = (f_output * output).sum(1)
        
#        output = torch.cat((output,features), dim=1)
        output = self.fc(output).squeeze()

        return output, h_output


class HierAttNet(nn.Module):
    def __init__(self, hidden_size, batch_size, num_classes, max_sent_len, dict_size, diction):
        super (HierAttNet, self).__init__()

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.sent_att_net = SentAttNet(self.hidden_size, num_classes)
        self.word_att_net_text = WordAttNet(dict_size, diction, hidden_size)

        self.max_sent_len = max_sent_len

        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.hidden_size)
        self.word_hidden_state = self.word_hidden_state.to(DEVICE)
        self.sent_hidden_state = self.sent_hidden_state.to(DEVICE)

    def forward(self, inputs, lengths, features, titles, title_lengths):
        # inputs = (B, S, W)
        text = inputs.permute(1, 0, 2)
        all_word_lengths = []
        output_list_text = []

#        import pdb; pdb.set_trace()

        for i in text:

            word_lengths = i.size(1) - (i==0).sum(dim=1)
            if 0 in word_lengths:
                for k in range(0, inputs.size()[0]):
                    if word_lengths[k] == 0:
                        word_lengths[k] = 1
#                all_word_lengths.append(word_lengths)
            output_text, self.word_hidden_state = self.word_att_net_text(i, word_lengths, 
									self.word_hidden_state, 
									is_title=False) #[8,600]
            output_list_text.append(output_text)
            self.word_hidden_state = repackage_hidden(self.word_hidden_state)



        output_title, self.word_hidden_state = self.word_att_net_text(titles,	### title
								title_lengths,
								self.word_hidden_state, 
								is_title=True) #[8,600]
        self.word_hidden_state = repackage_hidden(self.word_hidden_state)

        # output_list_text = (S, B, 600)
        # output_list_text = (S, B, 760)
#        import pdb; pdb.set_trace()
        
        output_list_text = pad_sequence(output_list_text, padding_len=self.batch_size)
        output, self.sent_hidden_state = self.sent_att_net(output_list_text, features, lengths, output_title, self.sent_hidden_state)
        self.sent_hidden_state = repackage_hidden(self.sent_hidden_state)
        return output


