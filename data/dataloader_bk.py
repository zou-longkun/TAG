import os
import glob
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from plyfile import PlyData
from utils.pc_utlis_sd import *
from utils.data_utils import *

NUM_POINTS = 1024
label_to_idx = {"bathtub": 0, "bed": 1, "bookshelf": 2, "cabinet": 3,
                "chair": 4, "lamp": 5, "monitor": 6,
                "plant": 7, "sofa": 8, "table": 9}


def read_ply(filename):
    """ read cordinates, return n * 3 """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


def process_data(pc):
    rand_points = np.random.uniform(-1, 1, 40000)
    x1 = rand_points[:20000]
    x2 = rand_points[20000:]
    power_sum = x1 ** 2 + x2 ** 2
    p_filter = power_sum < 1
    power_sum = power_sum[p_filter]
    sqrt_sum = np.sqrt(1 - power_sum)
    x1 = x1[p_filter]
    x2 = x2[p_filter]
    x = (2 * x1 * sqrt_sum).reshape(-1, 1)
    y = (2 * x2 * sqrt_sum).reshape(-1, 1)
    z = (1 - 2 * power_sum).reshape(-1, 1)
    density_points = np.hstack([x, y, z])
    fn = [
        lambda pc: drop_hole(pc, p=0.24),
        lambda pc: drop_hole(pc, p=0.36),
        lambda pc: drop_hole(pc, p=0.45),
        lambda pc: p_scan(pc, pixel_size=0.017),
        lambda pc: p_scan(pc, pixel_size=0.022),
        lambda pc: p_scan(pc, pixel_size=0.035),
        lambda pc: density(pc, density_points[np.random.choice(density_points.shape[0])], 1.3),
        lambda pc: density(pc, density_points[np.random.choice(density_points.shape[0])], 1.4),
        lambda pc: density(pc, density_points[np.random.choice(density_points.shape[0])], 1.6),
        lambda pc: pc.copy(),
    ]
    fn_index = list(range(len(fn)))
    ind = np.random.choice(fn_index)
    pc = fn[ind](pc)

    return pc


class ModelNet(Dataset):
    """
    modelnet dataset for pytorch dataloader
    """

    def __init__(self, args, partition='train'):
        self.args = args
        self.partition = partition
        self.pc_list = []
        self.data_list = []
        self.lbl_list = []
        PC_DIR = os.path.join(args.dataroot, "modelnet")
        DATA_DIR = os.path.join(args.dataroot, "modelnet_pro")

        data_dir_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', partition, '*.xyz')))
        pc_dir_list = sorted(glob.glob(os.path.join(PC_DIR, '*', partition, '*.npy')))

        for data_dir, pc_dir in zip(data_dir_list, pc_dir_list):
            self.pc_list.append(pc_dir)
            self.data_list.append(data_dir)
            self.lbl_list.append(label_to_idx[data_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.data_list)
        self.transforms = transforms.Compose([
            PointcloudToTensor(),
            PointcloudScale(),
            PointcloudRotatePerturbation(),
            PointcloudTranslate(),
            # PointcloudJitter(),
        ])

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

        print("number of " + partition + " examples in modelnet_pro : " + str(len(self.pc_list)))
        self.unique, self.counts = np.unique(self.label, return_counts=True)
        print("Occurrences count of classes in modelnet_pro " + partition + " set: " + str(
            dict(zip(self.unique, self.counts))))

    def __getitem__(self, item):
        pointcloud = np.load(self.pc_list[item])[:, :3].astype(np.float32)
        # pointcloud, furthest_distance = scale_to_unit_cube(pointcloud)
        pointcloud = normalization(pointcloud).astype(np.float32)

        data = np.loadtxt(self.data_list[item])[:, :3].astype(np.float32)
        # data = data / furthest_distance
        direc = np.loadtxt(self.data_list[item])[:, 3:6].astype(np.float32)
        direc = direc / np.linalg.norm(direc)
        dist = np.loadtxt(self.data_list[item])[:, 6].astype(np.float32)
        # dist = dist / furthest_distance
        label = np.copy(self.label[item])

        data_dic = {'cloud_ori': pointcloud, 'input': data, 'direc': direc, 'dist': dist, 'label': label}
        data_dic = build_correspondence(data_dic, self.args.seed_num, self.args.patch_size, relative=False)

        if self.args.is_aug:
            pointcloud = process_data(pointcloud)
            pointcloud = self.transforms(pointcloud).numpy()

        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        if pointcloud.shape[0] < NUM_POINTS:
            indices = np.random.randint(pointcloud.shape[0], size=(NUM_POINTS - pointcloud.shape[0]))
            tem_pointcloud = pointcloud[indices]
            tem_pointcloud = jitter_pointcloud(tem_pointcloud)
            pointcloud = np.concatenate((pointcloud, tem_pointcloud), axis=0)

        data_dic['cloud_aug'] = pointcloud

        data_dic = self.random_rotate_data_around_one_axis(data_dic, 'z')

        # structure = curvature_filter_pca(data_dic['cloud_aug'], str_pc_num=self.args.str_pc_num)
        # data_dic['structure'] = structure

        return data_dic

    def __len__(self):
        return len(self.data_list)

    def rotate_data(self, data, axis, angle):
        for key in ['cloud_ori', 'cloud_aug', 'input', 'patch', 'seeds', 'direc', 'dense_patch']:
            data[key] = rotate_one_axis_by_angle(data[key], axis, angle)
        return data

    def random_rotate_data_around_one_axis(self, data, axis):
        rotation_angle = np.random.uniform() * 2 * np.pi
        self.rotate_data(data, axis, rotation_angle)
        return data

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(10):
            cls_num_list.append(self.counts[i])
        return cls_num_list


class ShapeNet(Dataset):
    """
    shapenet dataset for pytorch dataloader
    """

    def __init__(self, args, partition='train'):
        self.args = args
        self.partition = partition
        self.pc_list = []
        self.data_list = []
        self.lbl_list = []
        PC_DIR = os.path.join(args.dataroot, "shapenet10_2048")
        DATA_DIR = os.path.join(args.dataroot, "shapenet10_2048_pro")

        data_dir_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', partition, '*.xyz')))
        pc_dir_list = sorted(glob.glob(os.path.join(PC_DIR, '*', partition, '*.npy')))

        for data_dir, pc_dir in zip(data_dir_list, pc_dir_list):
            self.pc_list.append(pc_dir)
            self.data_list.append(data_dir)
            self.lbl_list.append(label_to_idx[data_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.data_list)
        self.transforms = transforms.Compose([
            PointcloudToTensor(),
            PointcloudScale(),
            PointcloudRotatePerturbation(),
            PointcloudTranslate(),
            # PointcloudJitter(),
        ])

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

        print("number of " + partition + " examples in shapenet_pro : " + str(len(self.pc_list)))
        self.unique, self.counts = np.unique(self.label, return_counts=True)
        print("Occurrences count of classes in shapenet_pro " + partition + " set: " + str(
            dict(zip(self.unique, self.counts))))

    def __getitem__(self, item):
        pointcloud = np.load(self.pc_list[item])[:, :3].astype(np.float32)
        # pointcloud, furthest_distance = scale_to_unit_cube(pointcloud)
        pointcloud = normalization(pointcloud).astype(np.float32)

        data = np.loadtxt(self.data_list[item])[:, :3].astype(np.float32)
        # data = data / furthest_distance
        direc = np.loadtxt(self.data_list[item])[:, 3:6].astype(np.float32)
        direc = direc / np.linalg.norm(direc)
        dist = np.loadtxt(self.data_list[item])[:, 6].astype(np.float32)
        # dist = dist / furthest_distance
        label = np.copy(self.label[item])

        data_dic = {'cloud_ori': pointcloud, 'input': data, 'direc': direc, 'dist': dist, 'label': label}
        data_dic = build_correspondence(data_dic, self.args.seed_num, self.args.patch_size, relative=False)

        if self.args.is_aug:
            pointcloud = process_data(pointcloud)
            pointcloud = self.transforms(pointcloud).numpy()

        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        if pointcloud.shape[0] < NUM_POINTS:
            indices = np.random.randint(pointcloud.shape[0], size=(NUM_POINTS - pointcloud.shape[0]))
            tem_pointcloud = pointcloud[indices]
            tem_pointcloud = jitter_pointcloud(tem_pointcloud)
            pointcloud = np.concatenate((pointcloud, tem_pointcloud), axis=0)

        data_dic['cloud_aug'] = pointcloud

        data_dic = self.rotate_data(data_dic, 'x', -np.pi / 2)
        data_dic = self.random_rotate_data_around_one_axis(data_dic, 'z')

        # structure = curvature_filter_pca(data_dic['cloud_aug'], str_pc_num=self.args.str_pc_num)
        # data_dic['structure'] = structure

        return data_dic

    def __len__(self):
        return len(self.data_list)

    # shapenet is rotated such that the up direction is the z axis
    def rotate_data(self, data, axis, angle):
        for key in ['cloud_ori', 'cloud_aug', 'input', 'patch', 'seeds', 'direc', 'dense_patch']:
            data[key] = rotate_one_axis_by_angle(data[key], axis, angle)
        return data

    def random_rotate_data_around_one_axis(self, data, axis):
        rotation_angle = np.random.uniform() * 2 * np.pi
        self.rotate_data(data, axis, rotation_angle)
        return data

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(10):
            cls_num_list.append(self.counts[i])
        return cls_num_list


class ScanNet(Dataset):
    """
    scannet dataset for pytorch dataloader
    """

    def __init__(self, args, partition='train'):
        self.args = args
        self.partition = partition
        self.pc_list = []
        self.data_list = []
        self.lbl_list = []
        PC_DIR = os.path.join(args.dataroot, "scannet")
        DATA_DIR = os.path.join(args.dataroot, "scannet_pro")

        data_dir_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', partition, '*.xyz')))
        pc_dir_list = sorted(glob.glob(os.path.join(PC_DIR, '*', partition, '*.xyz')))

        for data_dir, pc_dir in zip(data_dir_list, pc_dir_list):
            self.pc_list.append(pc_dir)
            self.data_list.append(data_dir)
            self.lbl_list.append(label_to_idx[data_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.data_list)
        self.transforms = transforms.Compose([
            PointcloudToTensor(),
            # PointcloudScale(),
            # PointcloudRotatePerturbation(),
            PointcloudTranslate(),
            # PointcloudJitter(),
        ])

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

        print("number of " + partition + " examples in scannet_pro : " + str(len(self.pc_list)))
        self.unique, self.counts = np.unique(self.label, return_counts=True)
        print("Occurrences count of classes in scannet_pro " + partition + " set: " + str(
            dict(zip(self.unique, self.counts))))

    def __getitem__(self, item):
        pointcloud = np.loadtxt(self.pc_list[item])[:, :3].astype(np.float32)
        # pointcloud, furthest_distance = scale_to_unit_cube(pointcloud)
        pointcloud = normalization(pointcloud).astype(np.float32)

        data = np.loadtxt(self.data_list[item])[:, :3].astype(np.float32)
        # data = data / furthest_distance
        direc = np.loadtxt(self.data_list[item])[:, 3:6].astype(np.float32)
        direc = direc / np.linalg.norm(direc)
        dist = np.loadtxt(self.data_list[item])[:, 6].astype(np.float32)
        # dist = dist / furthest_distance
        label = np.copy(self.label[item])

        data_dic = {'cloud_ori': pointcloud, 'input': data, 'direc': direc, 'dist': dist, 'label': label}
        data_dic = build_correspondence(data_dic, self.args.seed_num, self.args.patch_size, relative=False)

        if self.args.is_aug:
            pointcloud = self.transforms(pointcloud).numpy()

        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        if pointcloud.shape[0] < NUM_POINTS:
            indices = np.random.randint(pointcloud.shape[0], size=(NUM_POINTS - pointcloud.shape[0]))
            tem_pointcloud = pointcloud[indices]
            tem_pointcloud = jitter_pointcloud(tem_pointcloud)
            pointcloud = np.concatenate((pointcloud, tem_pointcloud), axis=0)

        data_dic['cloud_aug'] = pointcloud

        data_dic = self.rotate_data(data_dic, 'x', -np.pi / 2)
        data_dic = self.random_rotate_data_around_one_axis(data_dic, 'z')

        # structure = curvature_filter_pca(data_dic['cloud_aug'], str_pc_num=self.args.str_pc_num)
        # data_dic['structure'] = structure

        return data_dic

    def __len__(self):
        return len(self.data_list)

    # scannet is rotated such that the up direction is the z axis
    def rotate_data(self, data, axis, angle):
        for key in ['cloud_ori', 'cloud_aug', 'input', 'patch', 'seeds', 'direc', 'dense_patch']:
            data[key] = rotate_one_axis_by_angle(data[key], axis, angle)
        return data

    def random_rotate_data_around_one_axis(self, data, axis):
        rotation_angle = np.random.uniform() * 2 * np.pi
        self.rotate_data(data, axis, rotation_angle)
        return data

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(10):
            cls_num_list.append(self.counts[i])
        return cls_num_list
