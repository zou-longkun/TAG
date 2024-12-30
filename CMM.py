import torch
import numpy as np
import numpy


class CrossManifoldMix:

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
        cloud_S = data_S.get('cloud').to(device).float()  # [batch_size, 2048, 3]
        cloud_T = data_T.get('cloud').to(device).float()  # [batch_size, 2048, 3]
        lam = np.random.beta(1.0, 1.0)

        fea_S = self.model.encoder(cloud_S)  # [bach_size, 2048]
        fea_T = self.model.encoder(cloud_T)  # [bach_size, 2048]
        fea_M = fea_S * lam + fea_T * (1 - lam)

        logits_S = self.model.classifier(fea_S)  # [bach_size, cls_num]
        logits_T = self.model.classifier(fea_T)  # [bach_size, cls_num]
        logits_M = self.model.classifier(fea_M)  # [bach_size, cls_num]

        loss = self.calc_loss(logits_S, logits_T, logits_M, lam)

        return loss

    def calc_loss(self, logits_S, logits_T, logits_M, lam):
        logits = logits_S * lam + (1 - lam) * logits_T
        loss = self.cos_sim(logits_M, logits)
        return loss




