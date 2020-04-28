import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from slp.modules.regularization import GaussianNoise

from slp.modules.attention import Attention
from slp.modules.embed import Embed
from slp.modules.helpers import PackSequence, PadPackedSequence

from slp.modules.util import pad_mask
from slp.modules.rnn import RNN
from slp.modules.feedforward import FF
from slp.modules.daclassifer import grad_reverse


class ConditionalEntropyLoss(nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)

class DACELoss(nn.Module):
    def __init__(self, loss_fn_cl, loss_fn_d, loss_fn_ce):
        super(DACELoss, self).__init__()
        self.loss_fn_cl = loss_fn_cl
        self.loss_fn_d = loss_fn_d
        self.loss_fn_ce = loss_fn_ce

    def forward(self, pred, tar, domain_pred, domain_targets, epoch):
        s_predictions = torch.stack([p for p,t in zip (pred, tar) if t>=0])
        s_targets = torch.stack([t for t in tar if t>=0])
        if -1 in tar: 
            t_predictions = torch.stack([p for p,t in zip(pred, tar) if t<0])
            loss_ce = self.loss_fn_ce(t_predictions)
        else:
            loss_ce = 0
        loss_cl = self.loss_fn_cl(s_predictions, s_targets)
        loss_d = self.loss_fn_d(domain_pred, domain_targets)
        return loss_cl + 0.01 * loss_d + 0.01 * loss_ce #NOTSURE

def switch_attr(m):
    if hasattr(m, 'track_running_stats'):
        m.track_running_stats ^= True
            

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VAT(nn.Module):

    def __init__(self, model, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VAT, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.model = model

    def forward(self, x, lengths):
        with torch.no_grad():
            p, _ = self.model(x, lengths)
            pred = F.softmax(p, dim=1)
        # prepare random unit tensor
        self.model.apply(switch_attr)
        sh = self.model.encoder.embed(x).shape
        d = torch.rand(sh).to(x.device)
        d = _l2_normalize(d)

        # calc adversarial direction
        for _ in range(self.ip):
            d.requires_grad_()
            pred_hat, _ = self.model(x, lengths, noise=True, d=d)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
            adv_distance.backward()
            d = _l2_normalize(d.grad)
            self.model.zero_grad()
    
        # calc LDS
        r_adv = d * self.eps
        pred_hat, _ = self.model(x, lengths, noise=True, d=r_adv)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        lds = F.kl_div(logp_hat, pred, reduction='batchmean')
        self.model.apply(switch_attr)
        return lds

class VADALoss(nn.Module):
    def __init__(self, loss_fn_cl, loss_fn_d, loss_fn_ce, loss_fn_vat):
        super(VADALoss, self).__init__()
        self.loss_fn_cl = loss_fn_cl
        self.loss_fn_d = loss_fn_d
        self.loss_fn_ce = loss_fn_ce
        self.loss_fn_vat = loss_fn_vat

    def forward(self, pred, tar, domain_pred, domain_targets, epoch, inputs, lengths):
        s_predictions = torch.stack([p for p,t in zip (pred, tar) if t>=0])
        s_targets = torch.stack([t for t in tar if t>=0])
        s_inputs = torch.stack([i for i,t in zip (inputs, tar) if t>=0])
        s_lengths = torch.stack([l for l,t in zip (lengths, tar) if t>=0])
        if -1 in tar: 
            t_predictions = torch.stack([p for p,t in zip(pred, tar) if t<0])
            t_inputs = torch.stack([i for i,t in zip (inputs, tar) if t<0])
            t_lengths = torch.stack([l for l,t in zip (lengths, tar) if t<0])
            loss_ce = self.loss_fn_ce(t_predictions)
            loss_vat_t = self.loss_fn_vat(t_inputs, t_lengths)
        else:
            loss_ce = 0
            loss_vat_t = 0
        loss_cl = self.loss_fn_cl(s_predictions, s_targets)
        loss_d = self.loss_fn_d(domain_pred, domain_targets)
        loss_vat_s = self.loss_fn_vat(s_inputs, s_lengths)
        #import ipdb; ipdb.set_trace()
        return loss_cl + 0.01 * loss_d + 0.01 * loss_ce + loss_vat_s + 0.01 * loss_vat_t #NOTSURE

class VADAClassifier(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes, num_domains):
        super(VADAClassifier, self).__init__()
        self.encoder = encoder
        self.clf = FF(encoded_features, num_classes,
                      activation='none', layer_norm=False,
                      dropout=0.)
        self.da = FF(encoded_features, num_domains,
                      activation='none', layer_norm=False,
                      dropout=0.)

    def forward(self, x, lengths, noise=False, d=None):
            x = self.encoder(x, lengths, noise, d)
            y = grad_reverse(x)
            return self.clf(x), self.da(y)

class VADAWordRNN(nn.Module):
    def __init__(
            self, hidden_size, embeddings,
            embeddings_dropout=.1, finetune_embeddings=False,
            batch_first=True, layers=1, bidirectional=False, merge_bi='cat',
            dropout=0.1, rnn_type='lstm', packed_sequence=True,
            attention=False, device='cpu'):
        super(VADAWordRNN, self).__init__()
        self.device = device
        self.embed = Embed(embeddings.shape[0],
                           embeddings.shape[1],
                           embeddings=embeddings,
                           dropout=embeddings_dropout,
                           trainable=finetune_embeddings)
        self.rnn = RNN(
            embeddings.shape[1], hidden_size,
            batch_first=batch_first, layers=layers, merge_bi=merge_bi,
            bidirectional=bidirectional, dropout=dropout,
            rnn_type=rnn_type, packed_sequence=packed_sequence)
        self.out_size = hidden_size if not bidirectional else 2 * hidden_size
        self.attention = None
        if attention:
            self.attention = Attention(
                attention_size=self.out_size, dropout=dropout)

    def forward(self, x, lengths, noise=False, d=None):
        x = self.embed(x)
        if noise:
            x = x + d 
        out, last_hidden, _ = self.rnn(x, lengths)
        if self.attention is not None:
            out, _ = self.attention(
                out, attention_mask=pad_mask(lengths, device=self.device))
            out = out.sum(1)
        else:
            out = last_hidden
        return out
