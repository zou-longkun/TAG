import numpy as np
import open3d as o3d
import open3d.core as o3c
from sklearn.neighbors import KDTree
import torch
import torch.utils.dlpack


def angle_axis(angle, axis):
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                              [u[2], 0.0, -u[0]],
                              [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    return R.float()


class PointcloudScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points


class PointcloudRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, points):
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = (
            points.new(points.size(0), 3)
            .normal_(mean=0.0, std=self.std)
            .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        translation = np.random.uniform(-self.translate_range, self.translate_range)
        points[:, 0:3] += translation
        return points


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


def normal_pc(pc):
    pc_mean = pc.mean(axis=0)
    pc = pc - pc_mean
    pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
    pc = pc/pc_L_max
    return pc


def density(pc, v_point=np.array([1, 0, 0]), gate=1):
    dist = np.sqrt((v_point ** 2).sum())
    max_dist = dist + 1
    min_dist = dist - 1
    dist = np.linalg.norm(pc - v_point.reshape(1,3), axis=1)
    dist = (dist - min_dist) / (max_dist - min_dist)
    r_list = np.random.uniform(0, 1, pc.shape[0])
    tmp_pc = pc[dist * gate < r_list]
    return tmp_pc


def p_scan(pc, pixel_size=0.017):
    pixel = int(2 / pixel_size)
    rotated_pc = rotate_point_cloud_3d(pc)
    pc_compress = (rotated_pc[:, 2] + 1) / 2 * pixel * pixel + (rotated_pc[:, 1] + 1) / 2 * pixel
    points_list = [None for i in range((pixel + 5) * (pixel + 5))]
    pc_compress = pc_compress.astype(np.int)
    for index, point in enumerate(rotated_pc):
        compress_index = pc_compress[index]
        if compress_index > len(points_list):
            print('out of index:', compress_index, len(points_list), point, pc[index], (pc[index] ** 2).sum(), (point ** 2).sum())
        if points_list[compress_index] is None:
            points_list[compress_index] = index
        elif point[0] > rotated_pc[points_list[compress_index]][0]:
            points_list[compress_index] = index
    points_list = list(filter(lambda x: x is not None, points_list))
    points_list = pc[points_list]
    return points_list


def drop_hole(pc, p):
    random_point = np.random.randint(0, pc.shape[0])
    index = np.linalg.norm(pc - pc[random_point].reshape(1, 3), axis=1).argsort()
    return pc[index[int(pc.shape[0] * p):]]


def rotate_point_cloud_3d(pc):
    rotation_angle = np.random.rand(3) * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix_1 = np.array([[cosval[0], 0, sinval[0]],
                                 [0, 1, 0],
                                 [-sinval[0], 0, cosval[0]]])
    rotation_matrix_2 = np.array([[1, 0, 0],
                                 [0, cosval[1], -sinval[1]],
                                 [0, sinval[1], cosval[1]]])
    rotation_matrix_3 = np.array([[cosval[2], -sinval[2], 0],
                                 [sinval[2], cosval[2], 0],
                                 [0, 0, 1]])
    rotation_matrix = np.matmul(np.matmul(rotation_matrix_1, rotation_matrix_2), rotation_matrix_3)
    rotated_data = np.dot(pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def vector_angle(x, y):
    Lx = np.sqrt(x.dot(x))
    Ly = (np.sum(y ** 2, axis=-1)) ** (0.5)
    cos_angle = np.sum(x * y, axis=-1) / (Lx * Ly)
    # assert cos_angle.any() > 1
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    return angle2


def don_filter_o3d(pc, str_pc_num=512):
    point_size = pc.shape[0]
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc)
    pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=300))
    normals_large = pcd1.normals

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc)
    pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals_small = pcd2.normals

    normal_angle = np.zeros(point_size)
    for i in range(point_size):
        normal_angle[i] = vector_angle(normals_large[i], normals_small[i])
    idx = np.argsort(normal_angle)[-str_pc_num:]
    point_don = pc[idx]
    freq_seg = np.zeros(point_size)
    freq_seg[idx] = 1
    return point_don.astype(np.float32), normal_angle.astype(np.float32), freq_seg.astype(np.float32)


def vector_angle_t(x, y):
    Lx = torch.sqrt(torch.dot(x, x))
    Ly = (torch.sum(y ** 2, dim=-1)) ** (0.5)
    cos_angle = torch.sum(x * y, dim=-1) / (Lx * Ly)
    cos_angle = torch.clip(cos_angle, -1, 1)
    angle = torch.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    return angle2


def don_filter_o3d_c(pc, str_pc_num=512):
    point_size = pc.shape[0]
    pc_o3c = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pc))
    pcd1 = o3d.t.geometry.PointCloud(pc_o3c.cpu())  # cuda(0): CUDA runtime error: API call is not supported in the installed CUDA driver
    pcd1.estimate_normals(max_nn=30, radius=1)
    # normals_large = torch.from_numpy(pcd1.point.normals.numpy())
    normals_large = torch.utils.dlpack.from_dlpack(pcd1.point.normals.to_dlpack())

    pcd2 = o3d.t.geometry.PointCloud(pc_o3c.cpu())
    pcd2.estimate_normals(max_nn=30, radius=0.1)
    # normals_small = torch.from_numpy(pcd2.point.normals.numpy())
    normals_small = torch.utils.dlpack.from_dlpack(pcd2.point.normals.to_dlpack())

    normal_angle = torch.zeros(point_size)
    for i in range(point_size):
        normal_angle[i] = vector_angle_t(normals_large[i], normals_small[i])
    idx = torch.argsort(normal_angle)[-str_pc_num:]
    point_don = pc[idx]
    freq_seg = torch.zeros(point_size)
    freq_seg[idx] = 1
    return point_don, normal_angle, freq_seg


def compute_norm_and_curvature(pc, knn_indices=None):
    if knn_indices is not None:
        pc = pc[knn_indices]
    covariance = np.cov(pc.T)
    w, v = np.linalg.eig(covariance)
    v = v.T
    w = np.real(w)
    i = np.argmin(np.abs(w))
    norm = v[i]
    curv = w[i] / (np.sum(np.abs(w)) + 1e-7)
    # assert curv is not complex
    return norm, np.real(curv)


def don_filter_pca(pc, k1=30, k2=5, str_pc_num=512):
    point_size = pc.shape[0]
    tree = KDTree(pc)
    dist1, idx1 = tree.query(pc, k1)
    dist2, idx2 = tree.query(pc, k2)
    normals1, normals2 = [], []
    for i in range(point_size):
        normal1, _ = compute_norm_and_curvature(pc, idx1[i])
        normal2, _ = compute_norm_and_curvature(pc, idx2[i])
        normals1.append(normal1)
        normals2.append(normal2)
    normal_angle = np.zeros(point_size)
    for i in range(point_size):
        normal_angle[i] = vector_angle(normals1[i], normals2[i])
    idx = np.argsort(normal_angle)[-str_pc_num:]
    point_don = pc[idx]
    freq_seg = np.zeros(point_size)
    freq_seg[idx] = 1
    return point_don.astype(np.float32), normal_angle.astype(np.float32), freq_seg.astype(np.float32)


def curvature_filter_pca(pc, k=30, str_pc_num=512):
    point_size = pc.shape[0]
    tree = KDTree(pc)
    dist, idx = tree.query(pc, k)
    curvs = np.zeros(point_size)
    for i in range(point_size):
        _, curv = compute_norm_and_curvature(pc, idx[i])
        curvs[i] = curv
    idx = np.argsort(curvs)[-str_pc_num:]
    point_don = pc[idx]
    freq_seg = np.zeros(point_size)
    freq_seg[idx] = 1
    return point_don.astype(np.float32), curvs.astype(np.float32), freq_seg.astype(np.float32)
