from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import copy
import sklearn.metrics as metrics
from utils.loss_utils import LabelSmoothingCrossEntropy, Distillation_Loss, Focal_Loss, AMSoftmaxLoss
from utils.emd import earth_mover_distance
from utils.data_utils import don_filter_o3d, don_filter_o3d_c
from utils.cross_block import CrossBlock
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Trainer:
    """ Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
    """

    def cos_sim(self, x1, x2):
        scores = torch.acos(torch.cosine_similarity(x1, x2, dim=1)) / numpy.pi
        return scores.mean()

    def __init__(self, model, model_pcg, optimizer, optimizer_ssl, optimizer_pcg, args, device=None):
        self.model = model
        self.model_pcg = model_pcg
        self.optimizer = optimizer
        self.optimizer_ssl = optimizer_ssl
        self.optimizer_pcg = optimizer_pcg
        self.args = args
        self.device = device
        self.alpha_teacher = 0.999
        self.crossblk = CrossBlock(2048, 8).to(device)

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            current = numpy.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(numpy.exp(-5.0 * phase * phase))

    def get_current_consistency_weight(self, current_epoch, init_weight, length):
        return init_weight * self.sigmoid_rampup(current_epoch, length)

    def train_step(self, data, global_step, current_epoch, domain='source'):
        """ Performs a training step.

        Args:
            data (dict): data dictionary
            :param args:
            :param current_epoch:
            :param global_step:
            :param domain:
            :param data:
        """
        self.model.train()
        self.model_pcg.train()

        # train model_pn
        self.optimizer.zero_grad()
        self.optimizer_ssl.zero_grad()
        output = self.compute_loss(data)
        # current_weight = self.get_current_consistency_weight(current_epoch, self.args.init_weight, self.args.epochs)
        if domain == 'source':
            loss = output['cls'] + output['ssl'] * 10. + output['str']  # + output['sim'] * 0.5  # output['align'] * 0.5
        else:
            loss = output['ssl'] * 10. + output['str']  # + output['sim'] * 0.5  # output['align'] * 0.1
        loss.backward()
        self.optimizer.step()
        self.optimizer_ssl.step()

        # train model_pcg
        self.optimizer_pcg.zero_grad()
        output = self.compute_loss(data)
        # current_weight = self.get_current_consistency_weight(current_epoch, self.args.init_weight, self.args.epochs)
        if domain == 'source':
            loss_pcg = output['cls_pcg'] + output['align_pcg'] * 0.5  # output['sim'] * 0.5
        else:
            loss_pcg = output['align_pcg'] * 0.5  # output['sim'] * 0.5
        loss_pcg.backward()
        self.optimizer_pcg.step()

        # scaler = torch.cuda.amp.GradScaler()
        # # train model_pn
        # self.optimizer.zero_grad()
        # self.optimizer_ssl.zero_grad()
        # with torch.cuda.amp.autocast():
        #     output = self.compute_loss(data)
        #     # current_weight = self.get_current_consistency_weight(current_epoch, self.args.init_weight, self.args.epochs)
        #     if domain == 'source':
        #         loss = output['cls'] + output['ssl'] * 10. + output['str']  # + output['sim'] * 0.5  # output['align'] * 0.5
        #     else:
        #         loss = output['ssl'] * 10. + output['str']  # + output['sim'] * 0.5  # output['align'] * 0.1
        # scaler.scale(loss).backward()
        # scaler.step(self.optimizer)
        # scaler.step(self.optimizer_ssl)
        # scaler.update()

        # # train model_pcg
        # self.optimizer_pcg.zero_grad()
        # with torch.cuda.amp.autocast():
        #     output = self.compute_loss(data)
        #     # current_weight = self.get_current_consistency_weight(current_epoch, self.args.init_weight, self.args.epochs)
        #     if domain == 'source':
        #         loss_pcg = output['cls_pcg'] + output['align_pcg'] * 0.5  # output['sim'] * 0.5
        #     else:
        #         loss_pcg = output['align_pcg'] * 0.5  # output['sim'] * 0.5
        # scaler.scale(loss_pcg).backward()
        # scaler.step(self.optimizer_pcg)
        # scaler.update()

        return loss.item() + loss_pcg.item()

    def evaluate(self, val_loader):
        """ Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        """

        pred_list = []
        true_list = []
        for data in tqdm(val_loader):
            output = self.eval_step(data)
            labels = output['labels']
            logits = output['logits']
            preds = logits.max(dim=1)[1]

            true_list.append(labels.cpu().numpy())
            pred_list.append(preds.detach().cpu().numpy())

        true = numpy.concatenate(true_list)
        pred = numpy.concatenate(pred_list)
        acc = metrics.accuracy_score(true, pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(true, pred)
        print("Evaluate - acc: %.4f, avg acc: %.4f" % (acc, avg_per_class_acc))

        return acc, avg_per_class_acc

    def evaluate_separate(self, val_loader):
        """ Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        """

        pred_list = []
        pred_pn_list = []
        pred_pcg_list = []
        pred_all_list = []
        pred_dpn_list = []
        pred_gpn_list = []
        true_list = []
        # for data in tqdm(val_loader):
        for data in val_loader:
            output = self.eval_step(data)
            labels = output['labels']
            logits, logits_pn, logits_pcg = output['logits']
            logits_all = logits + logits_pn + logits_pcg
            logits_dpn = logits + logits_pn
            logits_gpn = logits_pn + logits_pcg
            preds, preds_pn, preds_pcg = logits.max(dim=1)[1], logits_pn.max(dim=1)[1], logits_pcg.max(dim=1)[1]
            preds_all = logits_all.max(dim=1)[1]
            preds_dpn = logits_dpn.max(dim=1)[1]
            preds_gpn = logits_gpn.max(dim=1)[1]

            true_list.append(labels.cpu().numpy())
            pred_list.append(preds.detach().cpu().numpy())
            pred_pn_list.append(preds_pn.detach().cpu().numpy())
            pred_pcg_list.append(preds_pcg.detach().cpu().numpy())
            pred_all_list.append(preds_all.detach().cpu().numpy())
            pred_dpn_list.append(preds_dpn.detach().cpu().numpy())
            pred_gpn_list.append(preds_gpn.detach().cpu().numpy())

        true = numpy.concatenate(true_list)
        pred = numpy.concatenate(pred_list)
        pred_pn = numpy.concatenate(pred_pn_list)
        pred_pcg = numpy.concatenate(pred_pcg_list)
        pred_all = numpy.concatenate(pred_all_list)
        pred_dpn = numpy.concatenate(pred_dpn_list)
        pred_gpn = numpy.concatenate(pred_gpn_list)

        acc = metrics.accuracy_score(true, pred)
        acc_pn = metrics.accuracy_score(true, pred_pn)
        acc_pcg = metrics.accuracy_score(true, pred_pcg)
        acc_all = metrics.accuracy_score(true, pred_all)
        acc_dpn = metrics.accuracy_score(true, pred_dpn)
        acc_gpn = metrics.accuracy_score(true, pred_gpn)

        avg_per_class_acc = metrics.balanced_accuracy_score(true, pred)
        avg_per_class_acc_pn = metrics.balanced_accuracy_score(true, pred_pn)
        avg_per_class_acc_pcg = metrics.balanced_accuracy_score(true, pred_pcg)
        avg_per_class_acc_all = metrics.balanced_accuracy_score(true, pred_all)
        avg_per_class_acc_dpn = metrics.balanced_accuracy_score(true, pred_dpn)
        avg_per_class_acc_gpn = metrics.balanced_accuracy_score(true, pred_gpn)

        print("Evaluate - acc: %.4f, avg acc: %.4f" % (acc, avg_per_class_acc))
        print("Evaluate - acc_pn: %.4f, avg acc: %.4f" % (acc_pn, avg_per_class_acc_pn))
        print("Evaluate - acc_pcg: %.4f, avg acc: %.4f" % (acc_pcg, avg_per_class_acc_pcg))
        print("Evaluate - acc_all: %.4f, avg acc: %.4f" % (acc_all, avg_per_class_acc_all))
        print("Evaluate - acc_dpn: %.4f, avg acc: %.4f" % (acc_dpn, avg_per_class_acc_dpn))
        print("Evaluate - acc_gpn: %.4f, avg acc: %.4f" % (acc_gpn, avg_per_class_acc_gpn))

        return (acc, acc_pn, acc_pcg, acc_all, acc_dpn, acc_gpn), \
               (avg_per_class_acc, avg_per_class_acc_pn, avg_per_class_acc_pcg,
                avg_per_class_acc_all, avg_per_class_acc_dpn, avg_per_class_acc_gpn)

    def evaluate_separate_2(self, val_loader):
        """ Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        """

        pred_pn_list = []
        pred_pcg_list = []
        pred_gpn_list = []
        true_list = []
        # for data in tqdm(val_loader):
        for data in val_loader:
            output = self.eval_step(data)
            labels = output['labels']
            logits_pn, logits_pcg = output['logits']
            logits_gpn = logits_pn + logits_pcg
            preds_pn, preds_pcg = logits_pn.max(dim=1)[1], logits_pcg.max(dim=1)[1]
            preds_gpn = logits_gpn.max(dim=1)[1]

            true_list.append(labels.cpu().numpy())
            pred_pn_list.append(preds_pn.detach().cpu().numpy())
            pred_pcg_list.append(preds_pcg.detach().cpu().numpy())
            pred_gpn_list.append(preds_gpn.detach().cpu().numpy())

        true = numpy.concatenate(true_list)
        pred_pn = numpy.concatenate(pred_pn_list)
        pred_pcg = numpy.concatenate(pred_pcg_list)
        pred_gpn = numpy.concatenate(pred_gpn_list)

        acc_pn = metrics.accuracy_score(true, pred_pn)
        acc_pcg = metrics.accuracy_score(true, pred_pcg)
        acc_gpn = metrics.accuracy_score(true, pred_gpn)

        avg_per_class_acc_pn = metrics.balanced_accuracy_score(true, pred_pn)
        avg_per_class_acc_pcg = metrics.balanced_accuracy_score(true, pred_pcg)
        avg_per_class_acc_gpn = metrics.balanced_accuracy_score(true, pred_gpn)

        print("Evaluate - acc_pn: %.4f, avg acc: %.4f" % (acc_pn, avg_per_class_acc_pn))
        print("Evaluate - acc_pcg: %.4f, avg acc: %.4f" % (acc_pcg, avg_per_class_acc_pcg))
        print("Evaluate - acc_gpn: %.4f, avg acc: %.4f" % (acc_gpn, avg_per_class_acc_gpn))

        return (acc_pn, acc_pcg, acc_gpn), (avg_per_class_acc_pn, avg_per_class_acc_pcg, avg_per_class_acc_gpn)

    def eval_step(self, data):
        """ Performs an evaluation step.

        Args:
            data (dict): data dictionary
        """
        self.model.eval()
        self.model_pcg.eval()
        with torch.no_grad():
            output = self.compute_loss(data)

        return output

    def compute_loss(self, data):
        """ Computes the loss.

        Args:
            data (dict): data dictionary
        """
        device = self.device

        points = data.get('patch').to(device).float()  # [bach_size, 16, 256, 3]
        B, N, M, C = points.shape
        indices = numpy.random.randint(N, size=self.args.patch_num)
        points = points[:, indices, ...]
        points = points.reshape(-1, M, C)
        seeds = data.get('seeds').to(device).float()  # [batch_size, 16, 3]
        seeds = seeds[:, indices, ...]
        seeds = seeds.reshape(-1, 3)
        cloud = data.get('cloud_aug').to(device).float()  # [batch_size, 1024, 3]
        freq_gt = data.get('structure')[2].to(device).float()  # [batch_size, 512]

        direc = data.get('direc').to(device).float()  # [bach_size, 16, 3]
        direc = direc[:, indices, ...]
        dist = data.get('dist').to(device).float()  # [bach_size, 16]
        dist = dist[:, indices, ...]
        dist = torch.unsqueeze(dist, -1)  # [bach_size, 16, 1]
        ssl_gt = torch.cat((direc, dist), -1)  # [bach_size, 16, 4]
        ssl_gt = ssl_gt.reshape(-1)  # [bach_size * 16 * 4]
        cls_gt = data.get('label').to(device).long()  # [bach_size]
        local_cls_gt = cls_gt.unsqueeze(1).repeat(1, self.args.patch_num).reshape(-1).to(device).long()

        c_global_pn, xyz_list, c_list = self.model.encoder(cloud)  # [bach_size, 2048]
        global_logits_pn = self.model.classifier(c_global_pn)  # [bach_size, cls_num]
        freq_pn = self.model.decoder_str(xyz_list, c_list)

        # c_local = self.model.local_encoder(points)  # [bach_size * 16, 2048]
        _, c_local = self.model_pcg.encoder_dg(points)
        c_local_cat = torch.cat((c_local, seeds), 1)  # [bach_size * 16, 2051]
        n = self.model.decoder_geo(c_local_cat)  # [bach_size * 16, 4]
        n = n.reshape(-1)  # [bach_size * 16 * 4]

        c_local_cat = c_local_cat.reshape(B, self.args.patch_num, 2051, 1)
        c_local_cat = c_local_cat.permute(0, 2, 1, 3)
        c_local_pcg = self.model_pcg.pcg(c_local_cat)
        c_local_pcg = c_local_pcg.squeeze()
        pcg_logits = self.model_pcg.classifier_pcg(c_local_pcg)

        logits = (global_logits_pn, pcg_logits)

        criterion_sce = LabelSmoothingCrossEntropy()
        loss_cls = criterion_sce(global_logits_pn, cls_gt)
        loss_cls_pcg = criterion_sce(pcg_logits, cls_gt)

        loss_sim = self.cos_sim(c_local_pcg, c_global_pn)

        criterion_mse = torch.nn.MSELoss()
        loss_ssl = criterion_mse(n, ssl_gt)

        criterion_bce = torch.nn.BCEWithLogitsLoss()
        loss_str = criterion_bce(freq_pn, freq_gt)

        criterion_aign = nn.KLDivLoss(reduction="batchmean", log_target=True)
        loss_align_pcg = criterion_aign(F.log_softmax(pcg_logits, dim=1), F.log_softmax(global_logits_pn.detach(), dim=1))

        output = {'cls': loss_cls.float(), 'cls_pcg': loss_cls_pcg.float(), 'ssl': loss_ssl.float(),
                  'sim': loss_sim.float(), 'str': loss_str.float(), 'align': loss_align_pcg,
                  'align_pcg': loss_align_pcg, 'logits': logits, 'labels': cls_gt}

        return output
