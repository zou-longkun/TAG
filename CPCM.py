import torch
import numpy as np
from utils.pc_utlis import farthest_point_sample
import numpy
import sys
sys.path.append("./emd/")
import emd_module as emd


class CrossMix:

    def __init__(self, model, optimizer, device=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def cos_sim(self, x1, x2):
        scores = torch.acos(torch.cosine_similarity(x1, x2, dim=1)) / numpy.pi
        return scores.mean()

    def mixup_train_step(self, data_S, data_T):
        """ Performs a training step.

        Args:
            data (dict): data dictionary
            :param domain:
            :param data:
        """
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data_S, data_T)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_loss(self, data_S, data_T):
        """ Computes the loss.

        Args:
            data (dict): data dictionary
        """
        device = self.device
        cloud_S = data_S.get('cloud_aug').to(device).float()  # [batch_size, 2048, 3]
        cloud_T = data_T.get('cloud_aug').to(device).float()  # [batch_size, 2048, 3]
        batch_size = cloud_S.shape[0]
        index = torch.randperm(batch_size).to(device)
        cloud_M, lam = self.mix_shapes_r(cloud_S, cloud_T)
        # cloud_SM, lam_s = self.mix_shapes_r(cloud_S, cloud_S[index])
        # cloud_TM, lam_t = self.mix_shapes_r(cloud_T, cloud_T[index])

        fea_S, _, _ = self.model.encoder(cloud_S)  # [bach_size, 2048]
        fea_T, _, _ = self.model.encoder(cloud_T)  # [bach_size, 2048]
        fea_M, _, _ = self.model.encoder(cloud_M)  # [bach_size, 2048]
        # fea_SC = self.model.encoder(cloud_S[index])
        # fea_TC = self.model.encoder(cloud_T[index])
        # fea_SM = self.model.encoder(cloud_SM)
        # fea_TM = self.model.encoder(cloud_TM)

        logits_S = self.model.classifier(fea_S)  # [bach_size, cls_num]
        logits_T = self.model.classifier(fea_T)  # [bach_size, cls_num]
        logits_M = self.model.classifier(fea_M)  # [bach_size, cls_num]
        # logits_SC = self.model.classifier(fea_SC)
        # logits_TC = self.model.classifier(fea_TC)
        # logits_SM = self.model.classifier(fea_SM)
        # logits_TM = self.model.classifier(fea_TM)

        loss = self.calc_loss(logits_S, logits_T, logits_M, lam)  # + \
               # self.calc_loss(logits_S, logits_SC, logits_SM, lam_s) * 0.1 + \
               # self.calc_loss(logits_T, logits_TC, logits_TM, lam_t) * 0.1

        return loss

    def mix_shapes(self, X, Y):
        batch_size, num_points, _ = X.size()

        X = X.permute(0, 2, 1)
        Y = Y.permute(0, 2, 1)
        # draw lambda from beta distribution
        lam = np.random.beta(1.0, 1.0)

        num_pts_a = round(lam * num_points)
        num_pts_b = num_points - round(lam * num_points)

        _, pts_vals_a = farthest_point_sample(X, num_pts_a)
        _, pts_vals_b = farthest_point_sample(Y, num_pts_b)
        mixed_pc = torch.cat((pts_vals_a, pts_vals_b), 2)  # convex combination
        points_perm = torch.randperm(num_points).cuda()  # draw random permutation of points in the shape
        mixed_pc = mixed_pc[:, :, points_perm]
        mixed_pc = mixed_pc.permute(0, 2, 1)
        return mixed_pc, lam

    def mix_shapes_r(self, X, Y):
        batch_size, num_points, _ = X.size()

        # draw lambda from beta distribution
        lam = np.random.beta(1.0, 1.0)

        remd = emd.emdModule()
        remd = remd.cuda()
        dis, ind = remd(X, Y, 0.005, 300)
        for i1 in range(batch_size):
            Y[i1, :, :] = Y[i1, ind[i1].long(), :]

        int_lam = int(num_points * (1. - lam))
        int_lam = max(1, int_lam)
        gamma = np.random.choice(num_points, int_lam, replace=False, p=None)
        for i2 in range(batch_size):
            X[i2, gamma, :] = Y[i2, gamma, :]
        mixed_pc = X
        return mixed_pc, lam

    def mix_shapes_k(self, X, Y):
        batch_size, num_points, _ = X.size()

        # draw lambda from beta distribution
        lam = np.random.beta(1.0, 1.0)

        remd = emd.emdModule()
        remd = remd.cuda()
        dis, ind = remd(X, Y, 0.005, 300)
        for i1 in range(batch_size):
            Y[i1, :, :] = Y[i1, ind[i1].long(), :]

        int_lam = int(num_points * (1. - lam))
        int_lam = max(1, int_lam)
        random_point = torch.from_numpy(np.random.choice(num_points, batch_size, replace=False, p=None))
        # kNN
        ind1 = torch.tensor(range(batch_size))
        query = X[ind1, random_point].view(batch_size, 1, 3)
        dist = torch.sqrt(torch.sum((X - query.repeat(1, num_points, 1)) ** 2, 2))
        idxs = dist.topk(int_lam, dim=1, largest=False, sorted=True).indices
        for i2 in range(batch_size):
            X[i2, idxs[i2], :] = Y[i2, idxs[i2], :]
        mixed_pc = X
        return mixed_pc, lam

    def calc_loss(self, logits_S, logits_T, logits_M, lam):
        logits = logits_S * lam + (1 - lam) * logits_T
        loss = self.cos_sim(logits_M, logits)
        return loss
