import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.weight = weight

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * \
               F.nll_loss(log_preds, target, weight=self.weight, reduction=self.reduction)


class Distillation_Loss(nn.Module):
    def __init__(self, teacher_temp=0.1, student_temp=0.5):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        teacher_out = F.softmax((teacher_output / self.teacher_temp), dim=-1)
        total_loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)

        return total_loss.mean()


class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, num_classes=10, size_average=True):
        super(Focal_Loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        # preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30, weight=None):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))  # nj的四次开方
        m_list = m_list * (max_m / np.max(m_list))  # 常系数 C
        self.m_list = torch.FloatTensor(m_list).cuda()  # 转成 tensor
        assert s > 0
        self.s = s  # 这个参数的作用论文里提过么？
        self.weight = weight  # 和频率相关的 re-weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        index = torch.zeros_like(pred, dtype=torch.uint8)  # 和 x 维度一致全 0 的tensor
        index.scatter_(1, target.data.view(-1, 1), 1)  # dim idx input
        index_float = torch.FloatTensor(index.type).cuda()  # 转 tensor
        ''' 以上的idx指示的应该是一个batch的y_true '''
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        pred_m = pred - batch_m  # y 的 logit 减去 margin
        output = torch.where(index, pred_m, pred)  # 按照修改位置合并
        return self.ce(self.s * output, target, weight=self.weight)


class AMSoftmaxLoss(nn.Module):
    def __init__(self, in_feats=10, class_num=10, m=0.35, s=30, weight=None):
        super(AMSoftmaxLoss, self).__init__()
        self.in_feats = in_feats
        self.m = m
        self.s = s
        self.W = torch.nn.Parameter(torch.randn(in_feats, class_num), requires_grad=True)
        self.ce = nn.CrossEntropyLoss(weight=weight)
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, target):
        assert x.size()[0] == target.size()[0]
        assert x.size()[1] == self.in_feats
        device = x.device
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm).to(device)
        costh = torch.mm(x_norm, w_norm)
        target_view = target.view(-1, 1)
        delt_costh = torch.zeros(costh.size()).scatter_(1, target_view.cpu(), self.m).to(device)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        return self.ce(costh_m_s, target)



