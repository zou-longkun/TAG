import torch
import numpy
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import sklearn.metrics as metrics
from utils.pc_utlis import farthest_point_sample
from utils.loss_utils import LabelSmoothingCrossEntropy


class CrossGraphMix:

    def __init__(self, model, optimizer, device=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.classifier = class_classifier(2048, 11).to(device)

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
        output = self.compute_loss(data_S, data_T)
        loss = output['cls'] + output['mix']
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, val_loader):
        """ Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        """

        pred_list = []
        true_list = []
        self.model.eval()

        for data in tqdm(val_loader):
            labels = data.get('label')
            logits = self.eval_step(data)
            preds = logits.max(dim=1)[1]

            true_list.append(labels.cpu().numpy())
            pred_list.append(preds.detach().cpu().numpy())

        true = numpy.concatenate(true_list)
        pred = numpy.concatenate(pred_list)
        acc = metrics.accuracy_score(true, pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(true, pred)
        print("Evaluate - acc: %.4f, avg acc: %.4f" % (acc, avg_per_class_acc))

        return acc

    def eval_step(self, data):
        """ Performs an evaluation step.

        Args:
            data (dict): data dictionary
        """
        self.model.eval()
        with torch.no_grad():
            device = self.device
            points = data.get('input').to(device).float()  # [bach_size, 16, 100, 3]
            seeds = data.get('seeds').to(device).float()  # [batch_size, 16, 3]
            batch_size, part_num, _ = seeds.size()

            graph = self.model.encoder(points)  # [bach_size * 16, 2048]
            graph = graph.reshape(batch_size, part_num, -1)  # [bach_size, 16, 2048]
            graph = torch.cat((graph, seeds), 2)  # [bach_size, 16, 2051]
            graph = graph.reshape(batch_size, -1, 2051, 1)
            graph = graph.permute(0, 2, 1, 3)

            fea = self.model.pcg(graph).squeeze()
            logits = self.model.pcg_classifier(fea)  # [bach_size, cls_num]

        return logits

    def compute_loss(self, data_S, data_T):
        """ Computes the loss.

        Args:
            data (dict): data dictionary
        """
        device = self.device
        points_S = data_S.get('input').to(device).float()  # [bach_size, 16, 100, 3]
        points_T = data_S.get('input').to(device).float()  # [bach_size, 16, 100, 3]
        seeds_S = data_S.get('seeds').to(device).float()  # [batch_size, 16, 3]
        seeds_T = data_T.get('seeds').to(device).float()  # [batch_size, 16, 3]
        gt_S = data_S.get('label').to(device).long()  # [bach_size]
        batch_size, part_num, _ = seeds_S.size()

        graph_S = self.model.encoder(points_S)  # [bach_size * 16, 2048]
        graph_S = graph_S.reshape(batch_size, part_num, -1)  # [bach_size, 16, 2048]
        graph_S = torch.cat((graph_S, seeds_S), 2)  # [bach_size, 16, 2051]

        graph_T = self.model.encoder(points_T)  # [bach_size * 16, 2048]
        graph_T = graph_T.reshape(batch_size, part_num, -1)  # [bach_size, 16, 2048]
        graph_T = torch.cat((graph_T, seeds_T), 2)  # [bach_size, 16, 2051]
        graph_M, lam = self.mix_graph(graph_S, graph_T)

        graph_S = graph_S.reshape(batch_size, -1, 2051, 1)
        graph_S = graph_S.permute(0, 2, 1, 3)
        graph_T = graph_T.reshape(batch_size, -1, 2051, 1)
        graph_T = graph_T.permute(0, 2, 1, 3)
        graph_M = graph_M.reshape(batch_size, -1, 2051, 1)
        graph_M = graph_M.permute(0, 2, 1, 3)

        fea_S = self.model.pcg(graph_S).squeeze()
        fea_T = self.model.pcg(graph_T).squeeze()
        fea_M = self.model.pcg(graph_M).squeeze()

        logits_S = self.model.pcg_classifier(fea_S)  # [bach_size, cls_num]
        logits_T = self.model.pcg_classifier(fea_T)  # [bach_size, cls_num]
        logits_M = self.model.pcg_classifier(fea_M)  # [bach_size, cls_num]

        # logits_S = self.classifier(fea_S)  # [bach_size, cls_num]
        # logits_T = self.classifier(fea_T)  # [bach_size, cls_num]
        # logits_M = self.classifier(fea_M)  # [bach_size, cls_num]

        criterion_ce = torch.nn.CrossEntropyLoss()
        # criterion_ce = LabelSmoothingCrossEntropy()

        loss_mix = self.calc_loss(logits_S, logits_T, logits_M, lam)
        loss_cls = criterion_ce(logits_S, gt_S)

        output = {'cls': loss_cls.float(), 'mix': loss_mix.float(), 'logits': logits_S}

        return output

    def mix_graph(self, X, Y):
        batch_size, node_num, node_dim = X.size()  # [B,N,C]

        X = X.permute(0, 2, 1)
        Y = Y.permute(0, 2, 1)
        # draw lambda from beta distribution
        lam = np.random.beta(2.0, 2.0)

        num_nodes_a = round(lam * node_num)
        num_nodes_b = node_num - round(lam * node_num)

        sample_node_list = [i for i in range(node_num)]
        sampe_node_idx_a = random.sample(sample_node_list, num_nodes_a)
        subgraph_a = X[:, :, sampe_node_idx_a]
        sampe_node_idx_b = random.sample(sample_node_list, num_nodes_b)
        subgraph_b = Y[:, :, sampe_node_idx_b]

        # subgraph_a = self.generate_subgraph(X, num_nodes_a)
        # subgraph_b = self.generate_subgraph(Y, num_nodes_b)

        mixed_graph = torch.cat((subgraph_a, subgraph_b), 2)  # convex combination
        node_perm = torch.randperm(node_num).cuda()  # draw random permutation of nodes in the graph
        mixed_graph = mixed_graph[:, :, node_perm]
        mixed_graph = mixed_graph.permute(0, 2, 1)
        return mixed_graph, lam

    def calc_loss(self, logits_S, logits_T, logits_M, lam):
        logits = logits_S * lam + (1 - lam) * logits_T
        loss = self.cos_sim(logits_M, logits)
        return loss

    def generate_subgraph(self, X, k):
        batch_size, node_dim, node_num = X.size()  # [B,C,N]
        idx = self.knn(X, k)  # (batch_size, node_num, k)
        achor_id = torch.randint(node_num, size=(batch_size,))  # (batch_size,)
        idx = idx[torch.arange(batch_size), achor_id]  # (batch_size, k)
        idx = idx.unsqueeze(dim=1).repeat(1, node_dim, 1)  # (batch_size, node_dim, k)
        subgraph = torch.gather(X, dim=-1, index=idx)  # (batch_size, node_dim, k)
        return subgraph

    def knn(self, x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx


class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False, activation='relu', bias=True):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                nn.BatchNorm1d(out_ch),
                # nn.LayerNorm(out_ch),
                self.ac
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                self.ac
            )

    def forward(self, x):
        # x = l2_norm(x, 1)
        x = self.fc(x)
        return x


class class_classifier(nn.Module):
    def __init__(self, input_dim, num_class):
        super(class_classifier, self).__init__()

        self.mlp1 = fc_layer(input_dim, 512, bias=True, activation='leakyrelu', bn=True)
        self.dp1 = nn.Dropout(p=0.5)
        self.mlp2 = fc_layer(512, 256, bias=True, activation='leakyrelu', bn=True)
        self.dp2 = nn.Dropout(p=0.5)
        self.mlp3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.dp1(self.mlp1(x))
        x2 = self.dp2(self.mlp2(x))
        logits = self.mlp3(x2)
        return logits

