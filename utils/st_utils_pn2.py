import torch
import datetime
import math
import operator
import numpy as np
from collections import Counter
from utils.pc_utlis import random_rotate_one_axis
from torch.utils.data import Dataset
from SPCM import SourceMix
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import sklearn.metrics as metrics
from utils.loss_utils import LabelSmoothingCrossEntropy
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_progress(domain_set, partition, epoch, print_losses, true=None, pred=None):
    outstr = "%s - %s %d" % (partition, domain_set, epoch)
    acc = 0
    if true is not None and pred is not None:
        acc = metrics.accuracy_score(true, pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(true, pred)
        outstr += ", acc: %.4f, avg acc: %.4f" % (acc, avg_per_class_acc)

    for loss, loss_val in print_losses.items():
        outstr += ", %s loss: %.4f" % (loss, loss_val)
    datetime_string = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
    to_print = "%s: %s" % (datetime_string, outstr)
    print(to_print)
    return acc


def select_target_by_conf(trgt_train_loader, threshold, model=None):
    pc_list = []
    seed_list = []
    part_list = []
    direc_list = []
    dist_list = []
    pseudo_label_list = []
    gt_list = []
    sfm = nn.Softmax(dim=1)

    with torch.no_grad():
        model.eval()
        for data in trgt_train_loader:
            points = data.get('input').to(device).float()  # [bach_size, 128, 100, 3]
            seeds = data.get('seeds').to(device).float()  # [batch_size, 128, 3]
            cloud = data.get('cloud').to(device).float()  # [B, N, C]
            gt = data.get('label').to(device).long()
            direc = data.get('direc').to(device).float()  # [bach_size, 128, 3]
            dist = data.get('dist').to(device).float()  # [bach_size, 128]
            batch_size = cloud.shape[0]

            c_local = model.encoder(points)  # [bach_size * 128, 2048]
            local_logits = model.local_classifier(c_local)  # [bach_size * 128, 10]
            local_weights = torch.sigmoid(local_logits.max(-1)[0]).reshape(-1, 1)
            c_local = c_local * local_weights
            seeds_r = seeds.reshape(-1, 3)
            c_local_cat = torch.cat((c_local, seeds_r), 1)  # [bach_size * 128, 2051]
            c_local_cat = c_local_cat.reshape(batch_size, -1, 2051, 1)
            c_local_cat = c_local_cat.permute(0, 2, 1, 3)
            c_local_pcg = model.pcg(c_local_cat)
            c_local_pcg = c_local_pcg.squeeze()
            pcg_logits = model.pcg_classifier(c_local_pcg)

            c_global_pn = model.encoder_pn(cloud)  # [bach_size, 2048]
            global_logits_pn = model.classifier_pn(c_global_pn)  # [bach_size, cls_num]

            c_global = model.encoder(cloud)
            global_logits = model.classifier(c_global)

            logits = global_logits_pn + global_logits + pcg_logits

            cls_conf = sfm(logits)
            mask = torch.max(cls_conf, 1)  # 2 * b
            index = 0
            for i in mask[0]:
                if i > threshold:
                    pc_list.append(cloud[index].cpu().numpy())
                    seed_list.append(seeds[index].cpu().numpy())
                    part_list.append(points[index].cpu().numpy())
                    direc_list.append(direc[index].cpu().numpy())
                    dist_list.append(dist[index].cpu().numpy())
                    pseudo_label_list.append(mask[1][index].cpu().numpy())
                    gt_list.append(gt[index].cpu().numpy())
                index += 1
        print(len(pc_list))
        print('pseudo label acc: ', round(sum(np.array(pseudo_label_list) == np.array(gt_list)) / len(pc_list), 3))
    return np.array(pc_list), np.array(seed_list), np.array(part_list),  np.array(direc_list), np.array(dist_list), \
           np.array(pseudo_label_list), np.array(gt_list)


class DataLoadST(Dataset):
    def __init__(self, data, partition='train'):
        self.partition = partition
        self.pc, self.seeds, self.parts, self.direc, self.dist, self.label, self.gt_label = data
        self.num_examples = len(self.pc)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(int)
            np.random.shuffle(self.val_ind)

        print("number of examples in trgt_new_dataset: " + str(len(self.pc)))
        unique, counts = np.unique(self.label, return_counts=True)
        print("Occurrences count of classes in trgt_new_dataset set: " + str(dict(zip(unique, counts))))
        unique, counts = np.unique(self.gt_label, return_counts=True)
        print("Occurrences count of classes in trgt_new_gt_dataset set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.copy(self.pc[item])
        pointcloud = random_rotate_one_axis(pointcloud, "z")
        # pointcloud = pointcloud.transpose(1, 0)
        seeds = np.copy(self.seeds[item])
        parts = np.copy(self.parts[item])
        direc = np.copy(self.direc[item])
        dist = np.copy(self.dist[item])
        label = np.copy(self.label[item])
        data_dic = {'cloud': pointcloud, 'seeds': seeds, 'input': parts, 'direc': direc, 'dist': dist, 'label': label}
        return data_dic

    def __len__(self):
        return len(self.pc)

    def get_labels(self):
        return self.label


def self_train(train_loader, model, epochs=10):
    count = 0.0
    print_losses = {'cls': 0.0, 'pseudo_mix': 0.0}
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    scheduler = CosineAnnealingLR(opt, epochs)
    # criterion = nn.CrossEntropyLoss()  # return the mean of CE over the batch
    criterion = LabelSmoothingCrossEntropy()
    pseudo_mix_trainer = SourceMix(model, opt, device=device)

    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            opt.zero_grad()
            points = data.get('input').to(device).float()  # [bach_size, 128, 100, 3]
            seeds = data.get('seeds').to(device).float()  # [batch_size, 128, 3]
            cloud = data.get('cloud').to(device).float()  # [B, N, C]
            labels = data.get('label').to(device).long()
            batch_size = cloud.shape[0]

            c_local = model.encoder(points)  # [bach_size * 128, 2048]
            local_logits = model.local_classifier(c_local)  # [bach_size * 128, 10]
            local_weights = torch.sigmoid(local_logits.max(-1)[0]).reshape(-1, 1)
            c_local = c_local * local_weights
            seeds = seeds.reshape(-1, 3)
            c_local_cat = torch.cat((c_local, seeds), 1)  # [bach_size * 128, 2051]
            c_local_cat = c_local_cat.reshape(batch_size, -1, 2051, 1)
            c_local_cat = c_local_cat.permute(0, 2, 1, 3)
            c_local_pcg = model.pcg(c_local_cat)
            c_local_pcg = c_local_pcg.squeeze()
            pcg_logits = model.pcg_classifier(c_local_pcg)

            c_global_pn = model.encoder_pn(cloud)  # [bach_size, 2048]
            global_logits_pn = model.classifier_pn(c_global_pn)  # [bach_size, cls_num]

            c_global = model.encoder(cloud)
            global_logits = model.classifier(c_global)

            loss_cls = criterion(global_logits, labels) + criterion(global_logits_pn, labels) + criterion(pcg_logits, labels)
            # print(criterion(global_logits_pn, labels), criterion(global_logits, labels), criterion(pcg_logits, labels))
            print_losses['cls'] += loss_cls.item()
            loss_cls.backward()
            opt.step()

            pseudo_mix_loss = pseudo_mix_trainer.train_step(data)
            print_losses['pseudo_mix'] += pseudo_mix_loss

            count += batch_size

        scheduler.step()

        print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
        print_progress("Target_new", "Trn", epoch, print_losses)
