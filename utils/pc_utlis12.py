import torch
import random
from sklearn.neighbors import KDTree
import numpy as np


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device

    B, C, N = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # B x npoint
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    centroids_vals = torch.zeros(B, C, npoint).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].view(B, 3, 1)  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].clone()
        dist = torch.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[
            mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = torch.max(distance, -1)[1]  # get the index of the point farthest away
    return centroids, centroids_vals


def farthest_point_sample_np(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    B, C, N = xyz.shape
    centroids = np.zeros((B, npoint), dtype=np.int64)
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.randint(0, N, (B,), dtype=np.int64)
    batch_indices = np.arange(B, dtype=np.int64)
    centroids_vals = np.zeros((B, C, npoint))
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].reshape(B, 3, 1)  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].copy()
        dist = np.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[
            mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = np.argmax(distance, axis=1)  # get the index of the point farthest away
    return centroids, centroids_vals


def random_rotate_data_SO3(x1, x2, x3):
    """ Randomly rotate the point cloud to augument the dataset
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, rotated point cloud
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    x1 = rotate_one_axis_by_angle(x1, 'x', rotation_angle)
    x2 = rotate_one_axis_by_angle(x2, 'x', rotation_angle)
    x3 = rotate_one_axis_by_angle(x3, 'x', rotation_angle)

    rotation_angle = np.random.uniform() * 2 * np.pi
    x1 = rotate_one_axis_by_angle(x1, 'y', rotation_angle)
    x2 = rotate_one_axis_by_angle(x2, 'y', rotation_angle)
    x3 = rotate_one_axis_by_angle(x3, 'y', rotation_angle)

    rotation_angle = np.random.uniform() * 2 * np.pi
    x1 = rotate_one_axis_by_angle(x1, 'z', rotation_angle)
    x2 = rotate_one_axis_by_angle(x2, 'z', rotation_angle)
    x3 = rotate_one_axis_by_angle(x3, 'z', rotation_angle)

    return x1, x2, x3


def random_rotate_data_one_axis(x1, x2, x3, axis):
    rotation_angle = np.random.uniform() * 2 * np.pi
    x1 = rotate_one_axis_by_angle(x1, axis, rotation_angle)
    x2 = rotate_one_axis_by_angle(x2, axis, rotation_angle)
    x3 = rotate_one_axis_by_angle(x3, axis, rotation_angle)

    return x1, x2, x3


def rotate_one_axis_by_angle(x, axis, angle):
    """
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
        angle: rotation angle
    Return:
        A rotated shape
    """
    R_x = np.asarray([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    R_y = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    R_z = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    if axis == "x":
        return x.dot(R_x).astype('float32')
    elif axis == "y":
        return x.dot(R_y).astype('float32')
    else:
        return x.dot(R_z).astype('float32')


def random_rotate_one_axis(X, axis):
    """
    Apply random rotation about one axis
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
    Return:
        A rotated shape
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    if axis == 'x':
        R_x = [[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]]
        X = np.matmul(X, R_x)
    elif axis == 'y':
        R_y = [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        X = np.matmul(X, R_y)
    else:
        R_z = [[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]
        X = np.matmul(X, R_z)
    return X.astype('float32')


def translate_pointcloud(pointcloud):
    """
    Input:
        pointcloud: pointcloud data, [B, C, N]
    Return:
        A translated shape
    """
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    """
    Input:
        pointcloud: pointcloud data, [B, C, N]
        sigma:
        clip:
    Return:
        A jittered shape
    """
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud.astype('float32')


def scale_to_unit_cube(x):
    """
   Input:
       x: pointcloud data, [B, C, N]
   Return:
       A point cloud scaled to unit cube
   """
    if len(x) == 0:
        return x

    centroid = np.mean(x, axis=0)
    x -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(x) ** 2, axis=-1)))
    x /= furthest_distance
    return x, furthest_distance


def normalization(cloud):
    bbox = np.zeros((2, 3))
    bbox[0][0] = np.min(cloud[:, 0])
    bbox[0][1] = np.min(cloud[:, 1])
    bbox[0][2] = np.min(cloud[:, 2])
    bbox[1][0] = np.max(cloud[:, 0])
    bbox[1][1] = np.max(cloud[:, 1])
    bbox[1][2] = np.max(cloud[:, 2])
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()
    scale1 = 1 / scale
    for i in range(cloud.shape[0]):
        cloud[i] = cloud[i] - loc
        cloud[i] = cloud[i] * scale1
    return cloud


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    else:
        return np.eye(3)  # cross of all zeros only occurs on identical directions


def Subsampler(data, N, M, relative=True):
    data_out = data.copy()
    indices = np.random.randint(data['input'].shape[0], size=N)

    tlen = data_out['dist'][indices]
    tdirec = data_out['direc'][indices, :]
    tinput = data_out['input'][indices, :]
    tree = KDTree(data_out['cloud'])

    dist, idx = tree.query(tinput, M)
    data_out['input'] = data_out['cloud'][idx]
    data_out['seeds'] = tinput
    tinput = np.tile(np.expand_dims(tinput, 1), (1, M, 1))
    if relative:
        data_out['input'] = data_out['input'] - tinput
    data_out['direc'] = tdirec
    data_out['dist'] = tlen

    return data_out


def region_dropout_data(data, N, M):
    data_out = data.copy()
    indices = np.random.randint(data['input'].shape[0], size=N)

    region_centers = data_out['input'][indices, :]
    tree = KDTree(data['input'])
    dist, idx = tree.query(region_centers, M)

    data_out['input'] = np.delete(data['input'], idx, 0)
    data_out['dist'] = np.delete(data_out['dist'], idx, 0)
    data_out['direc'] = np.delete(data_out['direc'], idx, 0)

    return data_out


def region_dropout_pc(pc, N, M):
    data_out = pc.copy()
    indices = np.random.randint(pc.shape[0], size=N)

    region_centers = data_out[indices, :]
    tree = KDTree(pc)
    dist, idx = tree.query(region_centers, M)

    region_out = data_out[idx]
    data_out = np.delete(pc, idx, 0)

    return data_out, region_out


def assign_region_to_point_np(X, NREGIONS=3):
    """
    Input:
        X: point cloud [N, C]
    Return:
        Y: Region assignment per point [N]
    """

    n = NREGIONS
    d = 2 / n
    X_clip = np.clip(X, -0.99999999, 0.99999999)  # [N, C]
    num_points, _ = X.shape
    Y = np.zeros(num_points, dtype=np.int64)  # label matrix  [N]

    # The code below partitions all points in the shape to voxels.
    # At each iteration find per axis the lower threshold and the upper threshold values
    # of the range according to n (e.g., if n=3, then: -1, -1/3, 1/3, 1 - there are 3 ranges)
    # and save points in the corresponding voxel if they fall in the examined range for all axis.
    region_id = 0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                # lt= lower threshold, ut = upper threshold
                x_axis_lt = -1 + x * d < X_clip[:, 0]  # [N,]
                x_axis_ut = X_clip[:, 0] < -1 + (x + 1) * d  # [N,]
                y_axis_lt = -1 + y * d < X_clip[:, 1]  # [N,]
                y_axis_ut = X_clip[:, 1] < -1 + (y + 1) * d  # [N,]
                z_axis_lt = -1 + z * d < X_clip[:, 2]  # [N,]
                z_axis_ut = X_clip[:, 2] < -1 + (z + 1) * d  # [N,]
                # get a mask indicating for each coordinate of each point of each shape whether
                # it falls inside the current inspected ranges
                in_range = np.concatenate([x_axis_lt, x_axis_ut, y_axis_lt, y_axis_ut,
                                          z_axis_lt, z_axis_ut], axis=0)
                in_range = in_range.reshape(6, -1)  # [6, -1]
                # per each point decide if it falls in the current region only if in all
                # ranges the value is 1 (i.e., it falls inside all the inspected ranges)
                mask = np.min(in_range, axis=0)  # [N]
                Y[mask] = region_id  # label each point with the region id
                region_id += 1

    return Y


def remove_region_points_np(pc, NREGIONS=3):
    """
        Remove all points of a randomly selected region in the point cloud.
        Input:
            X - Point cloud [N, C]
        Return:
            X - Point cloud where points in a certain region are removed
        """
    # get points' regions
    data_out = pc.copy()
    regions = assign_region_to_point_np(pc)  # [N] N:the number of region_id
    n = NREGIONS
    region_ids = np.random.permutation(n ** 3)  # divide point cloud into 3**3 regions
    for i in region_ids:
        ind = regions == i  # [N]
        # if there are enough points in the region
        if 1024 >= np.sum(ind) >= 50:
            ind = np.argwhere(ind == True)
            # num_points = int(np.sum(ind))
            # rnd_ind = random.sample(range(0, x.shape[1]), num_points)
            # x[b, ind, :] = x[b, rnd_ind, :]
            region_out = data_out[ind]
            data_out = np.delete(pc, ind, 1)
            break  # move to the next shape in the batch
    return data_out, region_out


def voxel_fake_point_filter(point_cloud, leaf_size, select_num):
    x_min = y_min = z_min = -0.5
    x_max = y_max = z_max = 0.5

    Dx = int((x_max - x_min) // leaf_size + 1)
    Dy = int((y_max - y_min) // leaf_size + 1)
    Dz = int((z_max - z_min) // leaf_size + 1)

    R = np.ones(Dx * Dy * Dz)
    for i in range(len(point_cloud)):
        hx = int((point_cloud[i][0] - x_min) // leaf_size) - 1
        hy = int((point_cloud[i][1] - y_min) // leaf_size) - 1
        hz = int((point_cloud[i][2] - z_min) // leaf_size) - 1
        R[hx + hy * Dx + hz * Dx * Dy] = 0
        # neighboring face
        if hx - 1 + hy * Dx + hz * Dx * Dy > 0:
            R[hx - 1 + hy * Dx + hz * Dx * Dy] = 0
        if hx + 1 + hy * Dx + hz * Dx * Dy < (Dx * Dy * Dz):
            R[hx + 1 + hy * Dx + hz * Dx * Dy] = 0
        if hx + (hy - 1) * Dx + hz * Dx * Dy > 0:
            R[hx + (hy - 1) * Dx + hz * Dx * Dy] = 0
        if hx + (hy + 1) * Dx + hz * Dx * Dy < (Dx * Dy * Dz):
            R[hx + (hy + 1) * Dx + hz * Dx * Dy] = 0
        if hx + hy * Dx + (hz - 1) * Dx * Dy > 0:
            R[hx + hy * Dx + (hz - 1) * Dx * Dy] = 0
        if hx + hy * Dx + (hz + 1) * Dx * Dy < (Dx * Dy * Dz):
            R[hx + hy * Dx + (hz + 1) * Dx * Dy] = 0

        # neighboring edge
        if hx - 1 + (hy - 1) * Dx + hz * Dx * Dy > 0:
            R[hx - 1 + (hy - 1) * Dx + hz * Dx * Dy] = 0
        if hx + 1 + (hy - 1) * Dx + hz * Dx * Dy > 0:
            R[hx + 1 + (hy - 1) * Dx + hz * Dx * Dy] = 0
        if hx - 1 + (hy + 1) * Dx + hz * Dx * Dy < (Dx * Dy * Dz):
            R[hx - 1 + (hy + 1) * Dx + hz * Dx * Dy] = 0
        if hx + 1 + (hy + 1) * Dx + hz * Dx * Dy < (Dx * Dy * Dz):
            R[hx + 1 + (hy + 1) * Dx + hz * Dx * Dy] = 0

        if hx - 1 + hy * Dx + (hz - 1) * Dx * Dy > 0:
            R[hx - 1 + hy * Dx + (hz - 1) * Dx * Dy] = 0
        if hx + 1 + hy * Dx + (hz - 1) * Dx * Dy > 0:
            R[hx + 1 + hy * Dx + (hz - 1) * Dx * Dy] = 0
        if hx - 1 + hy * Dx + (hz + 1) * Dx * Dy < (Dx * Dy * Dz):
            R[hx - 1 + hy * Dx + (hz + 1) * Dx * Dy] = 0
        if hx + 1 + hy * Dx + (hz + 1) * Dx * Dy < (Dx * Dy * Dz):
            R[hx + 1 + hy * Dx + (hz + 1) * Dx * Dy] = 0

        if hx + (hy - 1) * Dx + (hz - 1) * Dx * Dy > 0:
            R[hx + (hy - 1) * Dx + (hz - 1) * Dx * Dy] = 0
        if hx + (hy + 1) * Dx + (hz - 1) * Dx * Dy > 0:
            R[hx + (hy + 1) * Dx + (hz - 1) * Dx * Dy] = 0
        if hx + (hy - 1) * Dx + (hz + 1) * Dx * Dy < (Dx * Dy * Dz):
            R[hx + (hy - 1) * Dx + (hz + 1) * Dx * Dy] = 0
        if hx + (hy + 1) * Dx + (hz + 1) * Dx * Dy < (Dx * Dy * Dz):
            R[hx + (hy + 1) * Dx + (hz + 1) * Dx * Dy] = 0

        # neighboring vertex
        if hx - 1 + (hy - 1) * Dx + (hz - 1) * Dx * Dy > 0:
            R[hx - 1 + (hy - 1) * Dx + (hz - 1) * Dx * Dy] = 0
        if hx + 1 + (hy - 1) * Dx + (hz - 1) * Dx * Dy > 0:
            R[hx + 1 + (hy - 1) * Dx + (hz - 1) * Dx * Dy] = 0
        if hx - 1 + (hy + 1) * Dx + (hz - 1) * Dx * Dy > 0:
            R[hx - 1 + (hy + 1) * Dx + (hz - 1) * Dx * Dy] = 0
        if hx + 1 + (hy + 1) * Dx + (hz - 1) * Dx * Dy > 0:
            R[hx + 1 + (hy + 1) * Dx + (hz - 1) * Dx * Dy] = 0
        if hx - 1 + (hy - 1) * Dx + (hz + 1) * Dx * Dy < (Dx * Dy * Dz):
            R[hx - 1 + (hy - 1) * Dx + (hz + 1) * Dx * Dy] = 0
        if hx + 1 + (hy - 1) * Dx + (hz + 1) * Dx * Dy < (Dx * Dy * Dz):
            R[hx + 1 + (hy - 1) * Dx + (hz + 1) * Dx * Dy] = 0
        if hx - 1 + (hy + 1) * Dx + (hz + 1) * Dx * Dy < (Dx * Dy * Dz):
            R[hx - 1 + (hy + 1) * Dx + (hz + 1) * Dx * Dy] = 0
        if hx + 1 + (hy + 1) * Dx + (hz + 1) * Dx * Dy < (Dx * Dy * Dz):
            R[hx + 1 + (hy + 1) * Dx + (hz + 1) * Dx * Dy] = 0

    f_num = sum(R == 1)
    f_idx = np.argwhere(R == 1)
    indices = np.random.randint(f_num, size=select_num)
    select_idx = f_idx[indices].squeeze()
    fake_pc = np.zeros((select_num, 3))
    for i, id in enumerate(select_idx):
        fake_pc[i][2] = - (id // (Dx * Dy) + 1) * leaf_size - z_min
        fake_pc[i][1] = - ((id % (Dx * Dy)) // Dx) * leaf_size - y_min
        fake_pc[i][0] = - ((id % (Dx * Dy)) % Dz) * leaf_size - x_min
    filtered_points = np.array(fake_pc, dtype=np.float64)
    filtered_points = jitter_pointcloud(filtered_points)

    return filtered_points