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


def pairs_cossim_matrix(v1, v2):
    # v.shape = (N, vector_dims)
    v1 = v1.permute(1, 0).unsqueeze(2).float()
    v2 = v2.permute(1, 0).unsqueeze(1).float()

    part1 = torch.matmul(v1, v2).sum(0)
    part2 = torch.matmul(v1.pow(2).sum(0).pow(0.5), v2.pow(2).sum(0).pow(0.5))

    return part1 / (part2 + 1e-15)


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


def knn(x, y, k):
    inner = -2 * torch.matmul(x, y.transpose(1, 0))
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(1, 0)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, k)
    return idx


def cossim_knn(x, y, k):
    scores = - torch.acos(pairs_cossim_matrix(x, y)) / np.pi
    idx = scores.topk(k=k, dim=-1)[1]
    return idx


def select_target_by_reliable_voting(src_train_loader, trgt_train_loader, model=None):
    src_fea_list = []
    src_label_list = []
    trgt_cloud_list = []
    trgt_fea_list = []
    trgt_label_list = []
    with torch.no_grad():
        model.eval()
        for data1, data2 in zip(src_train_loader, trgt_train_loader):
            src_cloud = data1.get('cloud').to(device).float()
            src_label = data1.get('label').long()
            src_c_global = model.encoder(src_cloud)
            trgt_cloud = data2.get('cloud').to(device).float()
            trgt_label = data2.get('label').long()
            trgt_c_global = model.encoder(trgt_cloud)
            src_fea_list.append(src_c_global.tolist())
            src_label_list.append(src_label.tolist())
            trgt_cloud_list.append(trgt_cloud.tolist())
            trgt_fea_list.append(trgt_c_global.tolist())
            trgt_label_list.append(trgt_label.tolist())
    src_fea_tensor = torch.tensor(sum(src_fea_list, []))
    src_label_arr = np.array(sum(src_label_list, []))
    trgt_cloud_arr = np.array(sum(trgt_cloud_list, []))
    trgt_fea_tensor = torch.tensor(sum(trgt_fea_list, []))
    trgt_label_arr = np.array(sum(trgt_label_list, []))
    idx = cossim_knn(trgt_fea_tensor, src_fea_tensor, 10)
    k_nearst_src_label_arr = src_label_arr[idx]
    mask = np.zeros(k_nearst_src_label_arr.shape[0])
    h = np.zeros([k_nearst_src_label_arr.shape[0], 2])
    for i, a in enumerate(k_nearst_src_label_arr):
        count = Counter(a).most_common(1)[0]
        h[i] = np.array(count)
        if count[1] == 10:
            mask[i] = 1
    print(sum(mask))
    mask = np.array(mask, dtype=bool)
    trgt_cloud_select_arr = trgt_cloud_arr[mask]
    trgt_label_select_arr = h[:, 0][mask].astype(np.int)
    trgt_label_gt_arr = trgt_label_arr[mask].astype(np.int)
    print('pseudo label acc: ', round(sum(trgt_label_select_arr == trgt_label_gt_arr) / sum(mask), 3))

    return trgt_cloud_select_arr, trgt_label_select_arr, trgt_label_gt_arr


def select_target_by_conf(trgt_train_loader, threshold, model=None):
    pc_list = []
    pseudo_label_list = []
    gt_list = []
    sfm = nn.Softmax(dim=1)

    with torch.no_grad():
        model.eval()
        for data in trgt_train_loader:
            cloud = data.get('cloud').to(device).float()
            gt = data.get('label')
            c_global = model.encoder(cloud)
            logits = model.classifier(c_global)
            cls_conf = sfm(logits)
            mask = torch.max(cls_conf, 1)  # 2 * b
            index = 0
            for i in mask[0]:
                if i > threshold:
                    pc_list.append(cloud[index].cpu().numpy())
                    pseudo_label_list.append(mask[1][index].cpu().numpy())
                    gt_list.append(gt[index])
                index += 1
        print(len(pc_list))
        print('pseudo label acc: ', round(sum(np.array(pseudo_label_list) == np.array(gt_list)) / len(pc_list), 3))
    return np.array(pc_list), np.array(pseudo_label_list), np.array(gt_list)


def select_target_by_clswise_conf(trgt_train_loader, threshold, model=None):
    pc_list = []
    gt_list = []
    pc_select_list = []
    label_select_list = []
    gt_select_list = []
    mask_val_list = []
    mask_ind_list = []
    sfm = nn.Softmax(dim=1)

    with torch.no_grad():
        model.eval()
        for data in trgt_train_loader:
            cloud = data.get('cloud').to(device).float()
            c_global = model.encoder(cloud)
            logits = model.classifier(c_global)
            cls_conf = sfm(logits)
            mask = torch.max(cls_conf, 1)  # 2 * b

            mask_val_list.extend(mask[0].cpu().numpy())  # confidence
            mask_ind_list.extend(mask[1].cpu().numpy())  # pseudo label

            pc_list.append(data['cloud'].tolist())
            gt_list.append(data['label'].tolist())

        mask_val_arr = np.array(mask_val_list)
        mask_ind_arr = np.array(mask_ind_list)
        pc_arr = np.array(sum(pc_list, []))
        gt_arr = np.array(sum(gt_list, []))
        mask_selct_by_threshold = mask_val_arr >= threshold

        mask_val_arr = mask_val_arr[mask_selct_by_threshold]
        mask_ind_arr = mask_ind_arr[mask_selct_by_threshold]
        pc_arr = pc_arr[mask_selct_by_threshold]
        gt_arr = gt_arr[mask_selct_by_threshold]

        sorted_id = np.argsort(mask_ind_arr)
        clswise_count = dict(sorted(Counter(mask_ind_arr).items(), key=operator.itemgetter(0), reverse=False))
        count_sum = sum(clswise_count.values())
        print(clswise_count, count_sum)
        start = 0
        for k, v in clswise_count.items():
            sorted_id_slice = sorted_id[start: start + v]
            mu = 1.0 - (v / count_sum)
            # print(mu)
            lenth = math.ceil(len(sorted_id_slice) * mu)

            mask_val_slice = mask_val_arr[sorted_id_slice]
            mask_val_select_slice_sorted_id = np.argsort(mask_val_slice)[:lenth]  # sorted by confidence

            pc_list_slice = pc_arr[sorted_id_slice]
            pc_list_select_slice = pc_list_slice[mask_val_select_slice_sorted_id]
            pc_select_list.append(pc_list_select_slice.tolist())

            gt_list_slice = gt_arr[sorted_id_slice]
            gt_list_select_slice = gt_list_slice[mask_val_select_slice_sorted_id]
            gt_select_list.append(gt_list_select_slice.tolist())

            mask_ind_slice = mask_ind_arr[sorted_id_slice]
            label_list_select_slice = mask_ind_slice[mask_val_select_slice_sorted_id]
            label_select_list.append(label_list_select_slice.tolist())

            start += v

        pc_select_arr = np.array(sum(pc_select_list, []))
        label_select_arr = np.array(sum(label_select_list, []))
        gt_select_arr = np.array(sum(gt_select_list, []))
        print(len(pc_select_arr))
        # print(len(label_select_arr))
        print('pseudo label acc: ', round(sum(label_select_arr == gt_select_arr) / len(pc_select_arr), 3))

    return pc_select_arr, label_select_arr, gt_select_arr


class DataLoadST(Dataset):
    def __init__(self, data):
        self.pc, self.label, self.gt_label = data
        self.num_examples = len(self.pc)

        print("number of examples in trgt_new_dataset: " + str(len(self.pc)))
        unique, counts = np.unique(self.label, return_counts=True)
        print("Occurrences count of classes in trgt_new_dataset set: " + str(dict(zip(unique, counts))))
        unique, counts = np.unique(self.gt_label, return_counts=True)
        print("Occurrences count of classes in trgt_new_gt_dataset set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.copy(self.pc[item])
        pointcloud = random_rotate_one_axis(pointcloud, "z")
        # pointcloud = pointcloud.transpose(1, 0)
        label = np.copy(self.label[item])
        data_dic = {'cloud': pointcloud, 'label': label}
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
            t_cloud = data.get('cloud').to(device).float()
            t_labels = data.get('label').to(device).long()
            batch_size = t_cloud.shape[0]

            pseudo_mix_loss = pseudo_mix_trainer.train_step(data)
            print_losses['pseudo_mix'] += pseudo_mix_loss * batch_size

            opt.zero_grad()
            t_c_global = model.encoder(t_cloud)
            t_logits = model.classifier(t_c_global)

            loss = criterion(t_logits, t_labels)
            print_losses['cls'] += loss.item() * batch_size
            loss.backward()
            count += batch_size
            opt.step()
        scheduler.step()

        print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
        print_progress("Target_new", "Trn", epoch, print_losses)
