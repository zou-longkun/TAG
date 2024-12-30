import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import random
import argparse
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchsampler import ImbalancedDatasetSampler
from trainer import Trainer
from SPCM import SourceMix
from CPCM import CrossMix
from utils.checkpoints import CheckpointIO
from model import Point_PN, Deep_PCG
from data.dataloader import ModelNet, ScanNet, ShapeNet
import multiprocessing


def split_set(dataset):
    """
    Input:
        dataset
        domain - modelnet/shapenet/scannet
        type_set - source/target
    output:
        train_sampler, valid_sampler
    """
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    unique, counts = np.unique(dataset.label[train_indices], return_counts=True)
    print("Occurrences count of classes in " +
          " train part: " + str(dict(zip(unique, counts))))
    unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
    print("Occurrences count of classes in " +
          " validation part: " + str(dict(zip(unique, counts))))
    # Creating PT data samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    train_sampler = ImbalancedDatasetSampler(dataset, labels=dataset.label[train_indices], indices=train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


def data(args):
    src_dataset = args.src_dataset
    trgt_dataset = args.trgt_dataset
    data_func = {'modelnet': ModelNet, 'scannet': ScanNet, 'shapenet': ShapeNet}

    src_train_dataset = data_func[src_dataset](args, 'train')
    src_test_dataset = data_func[src_dataset](args, 'test')
    trgt_train_dataset = data_func[trgt_dataset](args, 'train')
    trgt_test_dataset = data_func[trgt_dataset](args, 'test')

    src_train_sampler, src_valid_sampler = split_set(src_train_dataset)
    src_train_loader = torch.utils.data.DataLoader(src_train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.workers_num,
                                                   sampler=src_train_sampler, drop_last=True)
    src_val_loader = torch.utils.data.DataLoader(src_train_dataset, batch_size=args.batch_size,
                                                 num_workers=args.workers_num,
                                                 sampler=src_valid_sampler)
    src_test_loader = torch.utils.data.DataLoader(src_test_dataset, batch_size=args.test_batch_size,
                                                  num_workers=args.workers_num)

    trgt_train_sampler, trgt_valid_sampler = split_set(trgt_train_dataset)
    trgt_train_loader = torch.utils.data.DataLoader(trgt_train_dataset, batch_size=args.batch_size,
                                                    num_workers=args.workers_num,
                                                    sampler=trgt_train_sampler, drop_last=True)
    trgt_val_loader = torch.utils.data.DataLoader(trgt_train_dataset, batch_size=args.batch_size,
                                                  num_workers=args.workers_num,
                                                  sampler=trgt_valid_sampler)
    trgt_test_loader = torch.utils.data.DataLoader(trgt_test_dataset, batch_size=args.test_batch_size,
                                                   num_workers=args.workers_num)
    return src_train_loader, src_val_loader, src_test_loader, trgt_train_loader, trgt_val_loader, trgt_test_loader


def train(args, dataloader):
    # Arguments
    random.seed(1)
    # np.random.seed(1)  # to get the same point choice in ModelNet and ScanNet leave it fixed
    torch.manual_seed(args.random_seed)
    args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
    if args.cuda:
        print('Using GPUs ' + str(args.gpus) + ',' + ' from ' + str(torch.cuda.device_count()) + ' devices available')
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print('Using CPU')

    # Set t0
    t0 = time.time()
    src_train_loader, _, _, trgt_train_loader, _, trgt_test_loader = dataloader

    model = Point_PN(args).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ssl_module = nn.ModuleList([model.local_encoder, model.decoder_geo])
    optimizer_ssl = optim.Adam([{'params': ssl_module.parameters(), 'lr': args.lr_ssl}])

    ssl_params = list(map(id, ssl_module.parameters()))
    rest_params = filter(lambda x: id(x) not in ssl_params, model.parameters())
    optimizer = optim.Adam([{'params': rest_params, 'lr': args.lr}])
    scheduler = CosineAnnealingLR(optimizer, args.epochs)

    model_pcg = Deep_PCG(args).to(device)
    optimizer_pcg = optim.Adam(model_pcg.parameters(), lr=args.lr)
    scheduler_pcg = CosineAnnealingLR(optimizer_pcg, args.epochs)

    trainer = Trainer(model, model_pcg, optimizer, optimizer_ssl, optimizer_pcg, args, device=device)
    crossmix_trainer = CrossMix(model, optimizer, device=device)
    sourcemix_trainer = SourceMix(model, optimizer, device=device)
    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
    checkpoint_io_pcg = CheckpointIO(out_dir, model=model_pcg, optimizer=optimizer_pcg)

    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        load_dict = dict()
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)

    try:
        load_dict_pcg = checkpoint_io_pcg.load('model_pcg.pt')
    except FileExistsError:
        load_dict_pcg = dict()

    src_metric_val_best = 0.0

    # Shorthands
    nparameters = sum(p.numel() for p in model.parameters())
    logfile.write('Total number of parameters: %d\n' % nparameters)

    print_every = 50
    checkpoint_every = 200
    validate_every = 200

    while True:
        epoch_it += 1
        alpha = epoch_it / args.epochs
        logfile.flush()
        if epoch_it > args.epochs:
            logfile.close()
            break
        for src_batch, trgt_batch in zip(src_train_loader, trgt_train_loader):
            it += 1
            src_loss = trainer.train_step(src_batch, it, epoch_it, 'source')
            src_mix_loss = sourcemix_trainer.train_step(src_batch)
            trgt_loss = trainer.train_step(trgt_batch, it, epoch_it, 'target')
            logger.add_scalar('src_train/src_loss', src_loss, it)
            logger.add_scalar('src_mix_train/src_mix_loss', src_mix_loss, it)
            logger.add_scalar('trgt_train/trgt_loss', trgt_loss, it)

            # Mixup training
            crossmix_loss = crossmix_trainer.mixup_train_step(src_batch, trgt_batch)
            logger.add_scalar('crossmix_train/crossmix_loss', crossmix_loss, it)

            if print_every > 0 and (it % print_every) == 0 and it > 0:
                datetime_string = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
                logfile.write('%s - Source: [Epoch %02d] it=%03d, loss=%.6f, src_mix_loss=%.6f\n' %
                              (datetime_string, epoch_it, it, src_loss, src_mix_loss))
                text = 'Source: [Epoch %02d] it=%03d, loss=%.6f, src_mix_loss=%.6f' % (
                    epoch_it, it, src_loss, src_mix_loss)
                to_print = "%s: %s" % (datetime_string, text)
                print(to_print)

                datetime_string = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
                logfile.write('%s - Target: [Epoch %02d] it=%03d, loss=%.6f\n' %
                              (datetime_string, epoch_it, it, trgt_loss))
                text = 'Target: [Epoch %02d] it=%03d, loss=%.6f' % (epoch_it, it, trgt_loss)
                to_print = "%s: %s" % (datetime_string, text)
                print(to_print)

                datetime_string = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
                logfile.write('%s - Mixup: [Epoch %02d] it=%03d, loss=%.6f\n' %
                              (datetime_string, epoch_it, it, crossmix_loss))
                text = 'Mix-up: [Epoch %02d] it=%03d, loss=%.6f' % (epoch_it, it, crossmix_loss)
                to_print = "%s: %s" % (datetime_string, text)
                print(to_print)

            # Save checkpoint
            if (checkpoint_every > 0 and (it % checkpoint_every) == 0) and it > 0:
                logfile.write('Saving checkpoint\n')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it, acc_val_best=src_metric_val_best)
                checkpoint_io_pcg.save('model_pcg.pt')

            # Run validation
            if validate_every > 0 and (it % validate_every) == 0 and it > 0:
                # src_metric_val, src_metric_val_avg_acc = trainer.evaluate(src_val_loader)
                trgt_metric_test, trgt_metric_test_avg_acc = trainer.evaluate_separate_2(trgt_test_loader)
                # metric_val = metric_val.float()
                # logfile.write('Source validation metric: %.4f, avg acc: %.4f\n' %
                #               (src_metric_val, src_metric_val_avg_acc))
                # logfile.write('Target test metric: %.4f, avg acc: %.4f\n' %
                #               (trgt_metric_test, trgt_metric_test_avg_acc))
                for i in range(len(trgt_metric_test)):
                    logfile.write('Target test metric: %.4f, avg acc: %.4f\n' %
                                  (trgt_metric_test[i], trgt_metric_test_avg_acc[i]))
                # if src_metric_val > src_metric_val_best:
                #     src_metric_val_best = src_metric_val
                #     logfile.write('New best model (acc %.4f)\n' % src_metric_val_best)
                #     checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it, acc_val_best=src_metric_val_best)

        scheduler.step()
        scheduler_pcg.step()
        # print(optimizer_ssl.state_dict()['param_groups'][0]['lr'])
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        # print(optimizer_pcg.state_dict()['param_groups'][0]['lr'])

    logger.close()


def get_args():
    parser = argparse.ArgumentParser(description='DA on Point Clouds')
    parser.add_argument('--exp_name', type=str, default='myDA_new', help='Name of the experiment')
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

    # Shorthands
    out_dir = os.path.join(args.out_path, args.exp_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logfile_name = os.path.join(out_dir, 'log.txt')
    logfile = open(logfile_name, 'a')
    logger = SummaryWriter(os.path.join(out_dir, 'logs'))

    message = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    logfile.write(message)
    print(message)

    multiprocessing.set_start_method('spawn')
    dataloader = data(args)
    train(args, dataloader)


