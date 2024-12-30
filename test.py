import torch
import torch.optim as optim
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from trainer import Trainer
from utils.checkpoints import CheckpointIO
from model_bk import Point_PN, Deep_PCG
from data.dataloader import ModelNet, ScanNet, ShapeNet,label_to_idx
from utils.loss_utils import LabelSmoothingCrossEntropy
import sklearn.metrics as metrics
import argparse
import time


def test(test_loader, model=None):
    # Run on cpu or gpu
    count = 0.0
    print_losses = {'cls': 0.0}
    batch_idx = 0
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy()

    with torch.no_grad():
        model.eval()
        test_pred = []
        test_true = []
        start_time = time.time()
        for data in test_loader:
            cloud = data.get('cloud').to(device).float()
            labels = data.get('label').to(device).long()
            batch_size = cloud.shape[0]

            c_global = model.encoder(cloud)
            logits = model.classifier(c_global)
            loss = criterion(logits, labels)
            print_losses['cls'] += loss.item() * batch_size

            # evaluation metrics
            preds = logits.max(dim=1)[1]
            test_true.append(labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            count += batch_size
            batch_idx += 1
        end_time = time.time()
        print(f"Inference Time:{(end_time - start_time) / len(test_dataset) * 1000:.2f} ms")
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    conf_mat = metrics.confusion_matrix(test_true, test_pred, labels=list(label_to_idx.values())).astype(int)

    return conf_mat

def get_args():
    parser = argparse.ArgumentParser(description='DA on Point Clouds')
    parser.add_argument('--exp_name', type=str, default='test', help='Name of the experiment')
    parser.add_argument('--out_path', type=str, default='out', help='log folder path')
    parser.add_argument('--dataroot', type=str, default='../Sim2RealData/PointDA10', metavar='N', help='data path')
    parser.add_argument('--src_dataset', type=str, default='modelnet', choices=['modelnet', 'shapenet', 'scannet'])
    parser.add_argument('--trgt_dataset', type=str, default='scannet', choices=['modelnet', 'shapenet', 'scannet'])
    parser.add_argument('--is_aug', type=bool, default=True, help='whether dataaugmentation')
    parser.add_argument('--cls_num', type=int, default=10, help='number of categories')
    parser.add_argument('--patch_num', type=int, default=16, help='number of local patch')
    parser.add_argument('--patch_size', type=int, default=256, help='number of point per local patch')
    parser.add_argument('--seed_num', type=int, default=128, help='number of seed')
    parser.add_argument('--str_pc_num', type=int, default=512, help='number of high-frequency spatial structure points')
    parser.add_argument('--workers_num', type=int, default=8, help='number of workers')
    parser.add_argument('--epochs', type=int, default=400, help='number of episode to train')
    parser.add_argument('--random_seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                        help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of train batch per domain')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of test batch per domain')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_ssl', type=float, default=1e-4, help='learning rate of ssl')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--init_weight', type=float, default=1.0, help='init weight of distillation loss')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    # Arguments
    is_cuda = (torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Shorthands
    out_dir = 'out/test'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logfile = open('out/test/test_log.txt', 'a')
    batch_size = 16

    test_dataset = ScanNet(args, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    model = Point_PN(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # trainer = Trainer(model, optimizer, device=device)

    checkpoint_io = CheckpointIO('out/m2r', model=model, optimizer=optimizer)

    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        load_dict = dict()
    it = load_dict.get('it', -1)

    # Shorthands
    nparameters = sum(p.numel() for p in model.parameters())
    logfile.write('Total number of parameters: %d' % nparameters)

    # Run validation
    # metric_val = trainer.evaluate(test_loader)
    # metric_val = metric_val.float()
    # logfile.write('Validation metric : %.6f\n' % metric_val)

    trgt_conf_mat = test(test_loader, model)
    print("Test confusion matrix:")
    print('\n' + str(trgt_conf_mat))


