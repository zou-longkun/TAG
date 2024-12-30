import torch
import numpy as np
from utils.pc_utlis import farthest_point_sample
import sys
sys.path.append("./emd/")
import emd_module as emd


class SourceMix:
    """ Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
    """

    def __init__(self, model, optimizer, device=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_step(self, data):
        """ Performs a training step.

        Args:
            data (dict): data dictionary
            :param sourcemix:
            :param domain:
            :param data:
        """
        self.model.train()
        self.optimizer.zero_grad()
        output = self.compute_loss(data)
        loss = output['cls']
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_loss(self, data):
        """ Computes the loss.

        Args:
            data (dict): data dictionary
        """
        device = self.device
        cloud = data.get('cloud_aug').to(device).float()  # [batch_size, 2048, 3]
        cls_gt = data.get('label').to(device).long()  # [bach_size]

        # source mixup
        cloud, mix_gt = self.mix_shapes_r(cloud, cls_gt)

        c_global, _, _ = self.model.encoder(cloud)  # [bach_size, 2048]
        logits = self.model.classifier(c_global)  # [bach_size, cls_num]

        criterion_ce = torch.nn.CrossEntropyLoss()
        loss_cls = self.calc_mix_loss(logits, mix_gt, criterion_ce)

        output = {'cls': loss_cls.float()}

        return output

    def mix_shapes(self, X, Y):
        """
        combine 2 shapes arbitrarily in each batch.
        For more details see https://arxiv.org/pdf/2003.12641.pdf
        Input:
            X, Y - shape and corresponding labels
        Return:
            mixed shape, labels and proportion
        """
        X = X.permute(0, 2, 1)
        mixed_X = X.clone()
        batch_size, _, num_points = mixed_X.size()

        # uniform sampling of points from each shape
        device = X.device
        batch_size, _, num_points = X.size()
        index = torch.randperm(batch_size).to(device)  # random permutation of examples in batch

        # draw lambda from beta distribution
        lam = np.random.beta(2.0, 2.0)

        num_pts_a = round(lam * num_points)
        num_pts_b = num_points - round(lam * num_points)

        pts_indices_a, pts_vals_a = farthest_point_sample(X, num_pts_a)
        pts_indices_b, pts_vals_b = farthest_point_sample(X[index, :], num_pts_b)
        mixed_X = torch.cat((pts_vals_a, pts_vals_b), 2)  # convex combination
        points_perm = torch.randperm(num_points).to(device)  # draw random permutation of points in the shape
        mixed_X = mixed_X[:, :, points_perm]

        Y_a = Y.clone()
        Y_b = Y[index].clone()
        mixed_X = mixed_X.permute(0, 2, 1)

        return mixed_X, (Y_a, Y_b, lam)

    def mix_shapes_r(self, X, Y):
        batch_size, num_points, _ = X.size()
        device = X.device
        index = torch.randperm(batch_size).to(device)  # random permutation of examples in batch
        X1 = X
        X2 = X[index, :]

        # draw lambda from beta distribution
        lam = np.random.beta(2.0, 2.0)

        remd = emd.emdModule()
        remd = remd.cuda()
        dis, ind = remd(X1, X2, 0.005, 300)
        for i1 in range(batch_size):
            X2[i1, :, :] = X2[i1, ind[i1].long(), :]

        int_lam = int(num_points * (1. - lam))
        int_lam = max(1, int_lam)
        gamma = np.random.choice(num_points, int_lam, replace=False, p=None)
        for i2 in range(batch_size):
            X1[i2, gamma, :] = X2[i2, gamma, :]
        mixed_pc = X1
        Y_a = Y.clone()
        Y_b = Y[index].clone()
        return mixed_pc, (Y_a, Y_b, lam)

    def mix_shapes_k(self, X, Y):
        batch_size, num_points, _ = X.size()
        device = X.device
        index = torch.randperm(batch_size).to(device)  # random permutation of examples in batch
        X1 = X
        X2 = X[index, :]

        # draw lambda from beta distribution
        lam = np.random.beta(2.0, 2.0)

        remd = emd.emdModule()
        remd = remd.cuda()
        dis, ind = remd(X1, X2, 0.005, 300)
        for i1 in range(batch_size):
            X2[i1, :, :] = X2[i1, ind[i1].long(), :]

        int_lam = int(num_points * (1. - lam))
        int_lam = max(1, int_lam)
        random_point = torch.from_numpy(np.random.choice(num_points, batch_size, replace=False, p=None))
        # kNN
        ind1 = torch.tensor(range(batch_size))
        query = X1[ind1, random_point].view(batch_size, 1, 3)
        dist = torch.sqrt(torch.sum((X1 - query.repeat(1, num_points, 1)) ** 2, 2))
        idxs = dist.topk(int_lam, dim=1, largest=False, sorted=True).indices
        for i2 in range(batch_size):
            X1[i2, idxs[i2], :] = X2[i2, idxs[i2], :]
        mixed_pc = X1
        Y_a = Y.clone()
        Y_b = Y[index].clone()
        return mixed_pc, (Y_a, Y_b, lam)

    def calc_mix_loss(self, logits, mixup_vals, criterion):
        """
        Calculate loss between 2 shapes
        Input:
            logits
            mixup_vals: label of first shape, label of second shape and mixing coefficient
            criterion: loss function
        Return:
            loss
        """
        Y_a, Y_b, lam = mixup_vals
        loss = lam * criterion(logits, Y_a) + (1 - lam) * criterion(logits, Y_b)
        return loss * 0.5



