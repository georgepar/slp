import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from slp.modules.regularization import GaussianNoise

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

class VAT(nn.Module):
    def __init__(self, model):
        super(VAT, self).__init__()
        self.XI = 1e-6
        self.model = model
        self.epsilon = 3.5
        self.ng =  GaussianNoise(1)

    def forward(self, X, lengths, logit):
        vat_loss = self.virtual_adversarial_loss(X, logit, lengths)
        return vat_loss

    def generate_virtual_adversarial_perturbation(self, x, logit, lengths):
        x = x.float()
        d = Variable(x.data.new(x.size()).normal_(0, 1))
        d = self.XI * self.get_normalized_vector(d).requires_grad_()
        logit_m, _ = self.model((x + d).long(), lengths)
        dist = self.kl_divergence_with_logit(logit, logit_m)
        import ipdb; ipdb.set_trace()
        grad = torch.autograd.grad(dist, [d])[0] #not working
        d = grad.detach()
        return self.epsilon * self.get_normalized_vector(d)

    def kl_divergence_with_logit(self, q_logit, p_logit):
        q = F.softmax(q_logit, dim=1)
        qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
        qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
        return qlogq - qlogp

    def get_normalized_vector(self, d):
        return F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())

    def virtual_adversarial_loss(self, x, logit, lengths):
        r_vadv = self.generate_virtual_adversarial_perturbation(x, logit, lengths)
        logit_p = logit.detach()
        logit_m, _ = self.model(x + r_vadv, lengths)
        loss = self.kl_divergence_with_logit(logit_p, logit_m)
        return loss

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
            loss_vat_t = self.loss_fn_vat(t_inputs, t_lengths, t_predictions)
        else:
            loss_ce = 0
            loss_vat_t = 0
        loss_cl = self.loss_fn_cl(s_predictions, s_targets)
        loss_d = self.loss_fn_d(domain_pred, domain_targets)
        loss_vat_s = self.loss_fn_vat(s_inputs, s_lengths, s_predictions)
        return loss_cl + 0.01 * loss_d + 0.01 * loss_ce + loss_vat_s + 0.01 * loss_vat_t #NOTSURE