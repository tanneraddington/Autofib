#!/usr/bin/env python

from sklearn.decomposition import PCA
import numpy as np


import numpy as np
from sklearn.decomposition import PCA
import open3d as o3d
import pickle

def is_homogeneous_matrix(matrix):
    # Check matrix shape
    if matrix.shape != (4, 4):
        return False

    # Check last row
    if not np.allclose(matrix[3, :], [0, 0, 0, 1]):
        return False

    # Check rotational part (3x3 upper-left submatrix)
    rotational_matrix = matrix[:3, :3]
    if not np.allclose(np.dot(rotational_matrix, rotational_matrix.T), np.eye(3), atol=1.e-6) or \
            not np.isclose(np.linalg.det(rotational_matrix), 1.0, atol=1.e-6):
        
        print(np.linalg.inv(rotational_matrix), "\n")
        print(rotational_matrix.T)        
        print(np.linalg.det(rotational_matrix))
        
        return False

    return True


def find_pca_axes(obj_cloud, verbose=False):
    '''
    Given a point cloud determine a valid, right-handed coordinate frame
    '''
    pca_operator = PCA(n_components=3, svd_solver='full')
    pca_operator.fit(obj_cloud)
    centroid = np.matrix(pca_operator.mean_).T
    x_axis = pca_operator.components_[0]
    y_axis = pca_operator.components_[1]
    z_axis = np.cross(x_axis,y_axis)

    if verbose:
        print('PCA centroid', centroid)
        print('x_axis', x_axis)
        print('y_axis', y_axis)
        print('z_axis', z_axis)
    return np.array([x_axis, y_axis, z_axis]), centroid


#Compute angles between two vectors, code is from:
#https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def find_min_ang_vec(world_vec, cam_vecs):
    min_ang = float('inf')
    min_ang_idx = -1
    min_ang_vec = None
    for i in range(cam_vecs.shape[1]):
        angle = angle_between(world_vec, cam_vecs[:, i])
        larger_half_pi = False
        if angle > np.pi * 0.5:
            angle = np.pi - angle
            larger_half_pi = True
        if angle < min_ang:
            min_ang = angle
            min_ang_idx = i
            if larger_half_pi:
                min_ang_vec = -cam_vecs[:, i]
            else:
                min_ang_vec = cam_vecs[:, i]

    return min_ang_vec, min_ang_idx


def compute_world_to_object_frame_transformation(obj_cloud, verbose=True):
    '''
    For the given object cloud, build an object frame using PCA and aligning to the
    world frame.
    Returns a transformation from world frame to object frame.
    '''

    # Use PCA to find a starting object frame/centroid.
    axes, centroid = find_pca_axes(obj_cloud, verbose)
    axes = np.matrix(np.column_stack(axes))

    # Rotation from object frame to frame.
    R_o_w = np.eye(3)
    
    # x axes.
    x_axis = axes[:, 0]
    R_o_w[0, 0] = x_axis[0, 0]
    R_o_w[1, 0] = x_axis[1, 0]
    R_o_w[2, 0] = x_axis[2, 0]

    # y axes
    y_axis = axes[:, 1]
    R_o_w[0, 1] = y_axis[0, 0]
    R_o_w[1, 1] = y_axis[1, 0]
    R_o_w[2, 1] = y_axis[2, 0]

    # z axes
    z_axis = axes[:, 2]
    R_o_w[0, 2] = z_axis[0, 0]
    R_o_w[1, 2] = z_axis[1, 0]
    R_o_w[2, 2] = z_axis[2, 0]

    # Transpose to get rotation from world to object frame.
    R_w_o = np.transpose(R_o_w)
    d_w_o_o = np.dot(-R_w_o, centroid)
    
    # Build full transformation matrix.
    trans_matrix = np.eye(4)
    trans_matrix[:3,:3] = R_w_o
    trans_matrix[0,3] = d_w_o_o[0]
    trans_matrix[1,3] = d_w_o_o[1]
    trans_matrix[2,3] = d_w_o_o[2]    

    return trans_matrix



def compose_4x4_homo_mat(rotation, translation):
    ht_matrix = np.eye(4)
    ht_matrix[:3, :3] = rotation
    ht_matrix[:3, 3] = translation
    return ht_matrix


def rotate_around_z(ht_matrix, angle):
    rotation_z = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle),  np.cos(angle), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1]
    ])
    return np.dot(rotation_z, ht_matrix)


def invert_transformation_matrix(transformation_matrix):
    # Invert the rotation part by transposing the 3x3 top-left submatrix
    R_inv = transformation_matrix[:3, :3].T
    t = transformation_matrix[:3, 3]
    
    # Invert the translation part
    t_inv = -R_inv.dot(t)
    
    # Construct the inverted transformation matrix
    inverted_matrix = np.eye(4)
    inverted_matrix[:3, :3] = R_inv
    inverted_matrix[:3, 3] = t_inv
    
    return inverted_matrix


def transform_deformernet_action(action_translation_object, T_world_to_object):
    R_world_to_object = T_world_to_object[:3, :3]
    R_object_to_world = R_world_to_object.T # inverse of rotation matrix is its transpose
    assert action_translation_object.shape == (3,)

    action_translation_world = np.dot(R_object_to_world, action_translation_object)
  
    return action_translation_world  # shape (3,)


# def transform_deformernet_action(action_translation, action_rotation, T_world_to_object):
#     """
#     Inputs:
#     action_translation.shape: (3,)
#     action_rotation.shape: (3,3)
#     tf_matrix.shape: (4,4)

#     Output: transformed_action.shape: (4,4)
#     """
#     T_object_to_eef = compose_4x4_homo_mat(action_rotation, action_translation)  # shape (4,4)
#     modified_T_world_to_object = T_world_to_object.copy()
#     modified_T_world_to_object[:3,3] = 0

#     transformed_action = compute_transformed_action(modified_T_world_to_object, T_object_to_eef)

#     assert is_homogeneous_matrix(transformed_action)
#     return transformed_action


# def compute_transformed_action(T_world_to_object, T_object_to_eef):
#     # Compute the transformation matrix
#     T_world_to_eef = np.dot(T_world_to_object, T_object_to_eef)

#     return T_world_to_eef

# def transform_point_cloud(point_cloud, transformation_matrix):
#     # Add homogeneous coordinate (4th component) of 1 to each point
#     homogeneous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

#     # Apply the transformation matrix to each point
#     transformed_points = np.dot(homogeneous_points, transformation_matrix.T)

#     # Remove the homogeneous coordinate (4th component) from the transformed points
#     transformed_points = transformed_points[:, :3]

#     return transformed_points


def transform_point_cloud(point_cloud, transformation_matrix):
    """
    Transform a point cloud using a 4x4 transformation matrix.

    Parameters:
    - point_cloud (numpy.ndarray): Point cloud data with shape (N, 3).
    - transformation_matrix (numpy.ndarray): 4x4 transformation matrix.

    Returns:
    - numpy.ndarray: Transformed point cloud data with shape (N, 3).
    """
    # Add homogeneous coordinate (4th component) of 1 to each point
    homogeneous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

    # print("homogeneous_points.shape:", homogeneous_points.shape) # (N, 4)
    # print("transformation_matrix.shape:", transformation_matrix.shape) # (4, 4)

    # Apply the transformation matrix to each point
    transformed_points = np.dot(homogeneous_points, transformation_matrix.T)

    # Remove the homogeneous coordinate (4th component) from the transformed points
    transformed_points = transformed_points[:, :3]

    return transformed_points


def load_pickle_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def objectframeize_point_clouds(start_pc, goal_pc, start_point=None, goal_point=None):
    """
    Objectframeize point clouds

    takes in np arrays and returns np arrays in an objectframeized format
    """
    # Compute the mean of the tissue point cloudpr 
    tissue_mean = np.mean(start_pc, axis=0)

    # Shift all point clouds so that the tissue mean is at the origin
    start_pc -= tissue_mean
    goal_pc -= tissue_mean 
    if start_point is not None:
        start_point -= tissue_mean
    if goal_point is not None:
        goal_point -= tissue_mean

    # Perform PCA on the start point cloud
    pca = PCA(n_components=3)
    pca.fit(start_pc)

    # Create rotation matrix from PCA components
    rotation_matrix = pca.components_.T


    start = np.dot(start_pc, rotation_matrix)
    goal = np.dot(goal_pc, rotation_matrix)
    if start_point is not None:
        start_point = np.dot(start_point, rotation_matrix)
    if goal_point is not None:
        goal_point = np.dot(goal_point, rotation_matrix)
    displacement = None
    if start_point is not None and goal_point is not None:
        displacement = goal_point - start_point

    return start, goal, start_point, goal_point, displacement


def objectframeize_start_pc(start_pc):
    """
    Objectframeize point clouds

    takes in np arrays and returns np arrays in an objectframeized format
    """
    # Compute the mean of the tissue point cloudpr 
    tissue_mean = np.mean(start_pc, axis=0)

    # Shift all point clouds so that the tissue mean is at the origin
    start_pc -= tissue_mean

    # Perform PCA on the start point cloud
    pca = PCA(n_components=3)
    pca.fit(start_pc)

    # Create rotation matrix from PCA components
    rotation_matrix = pca.components_.T

    # Flip the z-axis (camera frame opposite)


    start = np.dot(start_pc, rotation_matrix)

    return start, rotation_matrix, tissue_mean

def deobjectframeize_point(point, rotation_matrix, tissue_mean):
    """
    Transform a point from object frame back to global frame
    
    Args:
        point: numpy array of shape (3,) representing point in object frame
        rotation_matrix: numpy array of shape (3,3) from objectframeize_start_pc
        tissue_mean: numpy array of shape (3,) representing original tissue mean
        
    Returns:
        numpy array of shape (3,) representing point in global frame
    """
    # Rotate point back using transpose of rotation matrix
    point_rotated = np.dot(point, rotation_matrix.T)
    
    # Translate back using tissue mean
    point_global = point_rotated + tissue_mean
    
    return point_global

def objectframeize_goal_pc(goal_pc, rotation_matrix, tissue_mean):
    """
    Transform goal point cloud into object frame using same parameters as start point cloud
    
    Args:
        goal_pc: numpy array of shape (N,3) representing goal point cloud
        rotation_matrix: numpy array of shape (3,3) from objectframeize_start_pc
        tissue_mean: numpy array of shape (3,) representing tissue mean from start pc
        
    Returns:
        numpy array of shape (N,3) representing goal point cloud in object frame
    """
    # Shift using same tissue mean as start point cloud
    goal_pc -= tissue_mean
    
    # Rotate using same rotation matrix as start point cloud
    goal = np.dot(goal_pc, rotation_matrix)
    
    return goal





def visualize_demo(demo_data):
    # Create point clouds
    start_pcd = o3d.geometry.PointCloud()
    start_pcd.points = o3d.utility.Vector3dVector(demo_data["start_pointcloud"])
    start_pcd.paint_uniform_color([1, 0, 0])  # Red for start

    goal_pcd = o3d.geometry.PointCloud()
    goal_pcd.points = o3d.utility.Vector3dVector(demo_data["goal_pointcloud"]) 
    goal_pcd.paint_uniform_color([0, 1, 0])  # Green for goal

    # Create coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.001)

    # Create spheres for start and goal points
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    start_sphere.translate(np.array(demo_data["start_position"]).reshape(3,1))
    start_sphere.paint_uniform_color([1, 0, 0])  # Red

    goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    goal_sphere.translate((np.array(demo_data["start_position"]) + np.array(demo_data["displacement"])).reshape(3,1))
    goal_sphere.paint_uniform_color([0, 1, 0])  # Green

    # Visualize everything
    o3d.visualization.draw_geometries([start_pcd, goal_pcd, frame, start_sphere, goal_sphere])


def pcd_ize(pc, color=None, vis=True, point_size=0.05):
   """
   Convert point cloud numpy array to an open3d object (usually for visualization purpose).
   """
   pcd = o3d.geometry.PointCloud()
   pcd.points = o3d.utility.Vector3dVector(pc)
    
   if color is not None:
       if (isinstance(color, np.ndarray) and color.shape[0] == pc.shape[0]) or \
          (isinstance(color, list) and len(color) == len(pc)):
           pcd.colors = o3d.utility.Vector3dVector(color/255.)   # color has to be normalized to [0,1]
       else:   
           pcd.paint_uniform_color(color)
          
   if vis:
       o3d.visualization.draw_geometries([pcd], point_size=point_size)
      
   return pcd


def down_sampling(pc, num_pts=1024, return_indices=False):


   """
   Input:
       pc: point cloud data, [B, N, D] where B = num batches, N = num points, D = feature size (typically D=3)
       num_pts: number of samples
   Return:
       centroids (numpy.ndarray): sampled pointcloud index, [num_pts, D]
       pc (numpy.ndarray): down_sampled point cloud, [num_pts, D]
   """


   if pc.ndim == 2:
       # insert batch_size axis
       pc = np.copy(pc)[None, ...]


   B, N, D = pc.shape
   xyz = pc[:, :,:3]
   centroids = np.zeros((B, num_pts))
   distance = np.ones((B, N)) * 1e10
   farthest = np.random.uniform(low=0, high=N, size=(B,)).astype(np.int32)


   for i in range(num_pts):
       centroids[:, i] = farthest
       centroid = xyz[np.arange(0, B), farthest, :] # (B, D)
       centroid = np.expand_dims(centroid, axis=1) # (B, 1, D)
       dist = np.sum((xyz - centroid) ** 2, -1) # (B, N)
       mask = dist < distance
       distance[mask] = dist[mask]
       farthest = np.argmax(distance, -1) # (B,)


   pc = pc[np.arange(0, B).reshape(-1, 1), centroids.astype(np.int32), :]


   if return_indices:
       return pc.squeeze(), centroids.astype(np.int32)


   return pc.squeeze()


def objectframeize_point_cloud_difDef(tissue, context, flip_x=False):
    # Compute the mean of the tissue point cloud
    tissue_mean = np.mean(tissue, axis=0)
    
    tissue -= tissue_mean
    context -= tissue_mean

    # Perform PCA on the context point cloud
    pca = PCA(n_components=3)
    pca.fit(context)

    # Create rotation matrix from PCA components
    rotation_matrix = pca.components_.T

    # # Flip the z-axis (camera frame opposite)
    rotation_matrix[:, 2] = -rotation_matrix[:, 2]
    
    if flip_x:
        rotation_matrix[:, 0] = -rotation_matrix[:, 0]

    tissue = np.dot(tissue, rotation_matrix)
    context = np.dot(context, rotation_matrix)


        

    # this is to ensure that the goal is always on the same side of the tissue (down the trachea)

    # Scale all point clouds to unit size
    # Get max distance from origin across all point clouds
    max_dist = 0
    for points in [tissue, context]:
        if len(points) > 0:
            dist = np.max(np.linalg.norm(points, axis=1))
            max_dist = max(max_dist, dist)
    


    # # Scale all points by reciprocal of max distance
    # scale_factor = 1.0 / max_dist
    # tissue *= scale_factor
    # context *= scale_factor 

    return tissue, context, rotation_matrix, tissue_mean


def goal_to_camera_frame(goal, rotation_matrix, tissue_mean):
    # Rotate back using the transpose of the rotation matrix
    goal = np.dot(goal, rotation_matrix.T)

    # Shift back using the tissue mean
    goal = goal + tissue_mean

    return goal



def visualize_prediction(pc_start, pc_goal, pred_start, pred_disp):
    # Create point clouds
    start_pcd = o3d.geometry.PointCloud()
    start_pcd.points = o3d.utility.Vector3dVector(pc_start)
    start_pcd.paint_uniform_color([1, 0, 0])  # Red for start

    goal_pcd = o3d.geometry.PointCloud()
    goal_pcd.points = o3d.utility.Vector3dVector(pc_goal)
    goal_pcd.paint_uniform_color([0, 1, 0])  # Green for goal

    # Create coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.005)

    # Create spheres for predicted points
    pred_start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
    pred_start_sphere.translate(pred_start.reshape(3,1))
    pred_start_sphere.paint_uniform_color([1, 0, 0])  # red for predicted start

    pred_goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
    pred_goal_sphere.translate((pred_start + pred_disp).reshape(3,1))
    pred_goal_sphere.paint_uniform_color([0, 1, 0])  # green for predicted goal

    # Create spheres for ground truth points
    # gt_start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    # gt_start_sphere.translate(gt_start.reshape(3,1))
    # gt_start_sphere.paint_uniform_color([0.25, 0, 0])  # Dark red for ground truth start

    # gt_goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    # gt_goal_sphere.translate((gt_start + gt_disp).reshape(3,1))
    # gt_goal_sphere.paint_uniform_color([0, 0.25, 0])  # Dark green for ground truth goal

    # Visualize everything
    o3d.visualization.draw_geometries([
        start_pcd, goal_pcd, frame,
        pred_start_sphere, pred_goal_sphere
    ])


def scale_point_clouds(pc1, pc2):
    """
    Scale point clouds to unit size and return scale factor.
    
    Args:
        pc1 (np.ndarray): First point cloud
        pc2 (np.ndarray): Second point cloud
        
    Returns:
        pc1_norm (np.ndarray): Normalized first point cloud
        pc2_norm (np.ndarray): Normalized second point cloud 
        scale_factor (float): Scale factor used for normalization
    """
    # # Get max distance from origin across both point clouds
    # max_dist = 0
    # for points in [pc1, pc2]:
    #     if len(points) > 0:
    #         dist = np.max(np.linalg.norm(points, axis=1))
    #         max_dist = max(max_dist, dist)

    # # Scale point clouds by reciprocal of max distance
    # scale_factor = 1.0 / max_dist
    # pc1_norm = pc1 * scale_factor
    # pc2_norm = pc2 * scale_factor

    # Scale point clouds by 1/1000
    scale_factor = 1/1000
    pc1_norm = pc1 * scale_factor
    pc2_norm = pc2 * scale_factor

    return pc1_norm, pc2_norm, scale_factor


def normalize_point_clouds(pc1, pc2):
    """
    Scale point clouds to unit size and return scale factor.
    
    Args:
        pc1 (np.ndarray): First point cloud
        pc2 (np.ndarray): Second point cloud
        
    Returns:
        pc1_norm (np.ndarray): Normalized first point cloud
        pc2_norm (np.ndarray): Normalized second point cloud 
        scale_factor (float): Scale factor used for normalization
    """
    # Get max distance from origin across both point clouds
    max_dist = 0
    for points in [pc1, pc2]:
        if len(points) > 0:
            dist = np.max(np.linalg.norm(points, axis=1))
            max_dist = max(max_dist, dist)

    # Scale point clouds by reciprocal of max distance
    scale_factor = 1.0 / max_dist
    pc1_norm = pc1 * scale_factor
    pc2_norm = pc2 * scale_factor


    return pc1_norm, pc2_norm, scale_factor



def filter_tumor_pc(tumor_pc):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tumor_pc)
    
    num_neighbors = 10
    std_ratio = 2.0 # points beyond this std are outliers
    pcd, _ = pcd.remove_statistical_outlier(num_neighbors, std_ratio)

    eps = 0.0005 # max distance to be in the same cluster
    min_points = 20 # min number of points in a cluster
    labels = np.array(pcd.cluster_dbscan(eps, min_points, print_progress=True))
    valid_points_mask = labels != -1
    tumor_pc = pcd.select_by_index(np.where(valid_points_mask)[0])
    tumor_pc = np.asarray(tumor_pc.points)

    # radii = [0.3, 0.6, 1.2]
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radii[0], max_nn=30))
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radii[0], radii[1], radii[2]]))
    return tumor_pc


def visualize_prediction_pcs(pc_start, pc_goal, pred_start, pred_disp):
    # Create point clouds
    start_pcd = o3d.geometry.PointCloud()
    start_pcd.points = o3d.utility.Vector3dVector(pc_start)
    start_pcd.paint_uniform_color([1, 0, 0])  # Red for start

    goal_pcd = o3d.geometry.PointCloud()
    goal_pcd.points = o3d.utility.Vector3dVector(pc_goal)
    goal_pcd.paint_uniform_color([0, 1, 0])  # Green for goal

    # Create coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Create spheres for predicted points
    pred_start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    pred_start_sphere.translate(pred_start.reshape(3,1))
    pred_start_sphere.paint_uniform_color([1, 0, 0])  # red for predicted start

    pred_goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    pred_goal_sphere.translate((pred_start + pred_disp).reshape(3,1))
    pred_goal_sphere.paint_uniform_color([0, 1, 0])  # green for predicted goal

    o3d.visualization.draw_geometries([
        start_pcd, goal_pcd
    ])

    # Visualize everything
    o3d.visualization.draw_geometries([
        start_pcd, goal_pcd, frame,
        pred_start_sphere, pred_goal_sphere
    ])

    use_push = input("Use this push? y/n: ")
    return use_push.lower() == "y"
