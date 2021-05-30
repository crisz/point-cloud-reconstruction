import open3d as o3d


def show_point_cloud(np_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    o3d.visualization.draw([pcd])
