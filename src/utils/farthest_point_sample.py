from pathlib import Path

import numpy as np

from utils.load_data import load_data
import open3d as o3
import random
import torch


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        fps-point: [npoint, D]
        centroids: [npoint] (indices)
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    centroids = centroids.astype(np.int32)
    point = np.array(point[centroids])
    return point, centroids


def random_occlude_pointcloud_v2(xyz, centroids=None, n_drop=200, n_c=1):
    """
    Drops 'n_drop' nearest neighboors around each of 'n_c' centroids
    Parameters
    :param xyz: input pointcloud
    :param centroids: if not None list of keypoints to drop around
    :param n_drop: number of nn to drop around each centroid
    :param n_c: number of centroids selected
    Returns
    :return: partial/cropped point cloud, missing part for completion GT
    """
    in_points = np.shape(xyz)[0]

    if n_drop <= 0:
        # do not crop shape - baseline with no crop augmentation
        return xyz, None

    if centroids is None:
        _, fps_idx = farthest_point_sample(xyz.numpy(), 10)
        centroids = xyz[fps_idx]

    pcd = o3.geometry.PointCloud()
    pcd.points = o3.utility.Vector3dVector(xyz)
    pcd_tree = o3.geometry.KDTreeFlann(pcd)

    assert len(centroids) >= n_c
    idxs = list(range(len(centroids)))
    random.shuffle(idxs)
    idxs = idxs[:n_c]  # taking only 'n_c' random centroids
    selected_centroids = centroids[idxs]

    dropped = []
    for curr in selected_centroids:
        k, idx, _ = pcd_tree.search_knn_vector_3d(curr, n_drop)  # pcd.points[idx], n_drop)
        dropped.extend(idx)

    accepted_idxs = np.asarray(list(set(list(range(in_points))) - set(dropped)))
    dropped = np.asarray(dropped)
    occluded_pointset = xyz[accepted_idxs]
    missing_part_gt = xyz[dropped]
    return occluded_pointset, missing_part_gt


def random_occlude_batch(batch, n_drop=200):
    batch = batch.cpu()
    bs, n_points, n_dim = batch.shape
    out_occluded = np.empty((bs, n_points-n_drop, n_dim))
    out_missing = np.empty((bs, n_drop, n_dim))
    for i in range(bs):
        out_occluded[i, :, :], out_missing[i, :, :] = random_occlude_pointcloud_v2(batch[i])

    out_occluded = torch.from_numpy(out_occluded).to(dtype=torch.float).cuda()
    out_missing = torch.from_numpy(out_missing).to(dtype=torch.float).cuda()
    return out_occluded, out_missing


def random_occlude_pointcloud_v3(xyz, centroids=None, scales=None, n_c=1):
    """
    @param xyz: pointcloud of size N,3
    @param centroids: list of centroids to crop around
    @param scales: [missing part size, missing part size + frame size]
    @param n_c: number of holes
    @return: partial point cloud to complete, missing part gt, missing part + frame gt
    """
    assert scales is not None
    assert len(scales) == 2  # missing + frame
    assert n_c >= 1
    assert len(centroids) >= n_c
    num_points = np.shape(xyz)[0]  # number of input points

    if centroids is None:
        print("None centroids")
        _, fps_idx = farthest_point_sample(xyz.numpy(), 10)
        centroids = xyz[fps_idx]

    pcd = o3.geometry.PointCloud()
    pcd.points = o3.utility.Vector3dVector(xyz)
    pcd_tree = o3.geometry.KDTreeFlann(pcd)

    idxs = list(range(len(centroids)))
    random.shuffle(idxs)
    idxs = idxs[:n_c]  # taking only 'n_c' random centroids
    chosen_centroids = centroids[idxs]

    # Scale 0: Missing Part
    dropped = []
    for curr in chosen_centroids:
        k, idx, _ = pcd_tree.search_knn_vector_3d(curr, scales[0])
        dropped.extend(idx)
    idx_accept = np.asarray(list(set(list(range(num_points))) - set(dropped)))
    partial = xyz[idx_accept]  # overall input shape - missing part
    missing_gt = xyz[np.asarray(dropped)]  # missing part GT

    # Scale 1: Missing Part + its frame points
    dropped_1 = []
    for curr in chosen_centroids:
        # scales[1] should be: #(missing) + #(context) where #(context) are the GT points for the Identity problem
        k, idx_1, _ = pcd_tree.search_knn_vector_3d(curr, scales[1])
        dropped_1.extend(idx_1)

    # idxs_accepted_1 = np.asarray(list(set(list(range(num_points))) - set(dropped_1)))
    # partial_1 = xyz[idxs_accepted_1]
    missing_gt_1 = xyz[np.asarray(dropped_1)]
    return partial, missing_gt, missing_gt_1  #, partial_1, missing_gt_1


if __name__ == '__main__':
    print("FPS 1.5")
    data = load_data(mode="train")
    data = np.delete(data, 3, axis=2)
    data = data[0]
    output1 = farthest_point_sample(data, 128)
    output2 = farthest_point_sample(data, 256)
    output3 = farthest_point_sample(data, 512)
    occluded = random_occlude_pointcloud_v2(torch.from_numpy(data))

    np.save(str(Path('') / 'test_fps.npy'), np.array([data, output1, output2, output3, occluded]))
    print("Saved")
