from scipy.spatial.transform import Rotation as R
import numpy as np

def calculate_relative_pose(real_pose, detected_pose, is_quat):
    if is_quat == False:
        real_pos, real_vec = real_pose
        detected_pos, detected_vec = detected_pose
        relative_rotation = rotation_matrix_between_vectors(detected_vec, real_vec)
        relative_pose = (real_pos - detected_pos, relative_rotation) # fake + relative = real; detected * relative = real
        return relative_pose
    else:
        real_pos, real_quat = real_pose
        detected_pos, detected_quat = detected_pose
        rr = quaternion_to_rotation_matrix(real_quat)
        dr = quaternion_to_rotation_matrix(detected_quat)
        # relative_rotation = np.linalg.inv(dr).dot(rr)
        relative_rotation = rr.dot(np.linalg.inv(dr))
        relative_pose = (real_pos - detected_pos, relative_rotation) # fake + relative = real; detected * relative = real
        return relative_pose

def get_real_r(relative_rotation, detected_rotation):
    # real_rotation = detected_rotation.dot(relative_rotation)
    real_rotation = relative_rotation.dot(detected_rotation)
    return real_rotation

def rotation_matrix_between_vectors(a, b):
    a = np.array(a)
    b = np.array(b)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    if (a == b).all():
        return np.eye(3)
    rotation = R.align_vectors([a], [b])
    return rotation[0].as_matrix()

def extract_z_axis(ori):
    x, y, z, w = ori
    rotation_matrix = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    gripper_local_z = np.array([0, 0, 1])
    gripper_z_base = rotation_matrix.dot(gripper_local_z)
    gripper_z_base /= np.linalg.norm(gripper_z_base)
    return gripper_z_base

def quaternion_to_rotation_matrix(quaternion):
    x, y, z, w = quaternion
    rotation_matrix = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return rotation_matrix

def r_to_quat(r):
    rotation = R.from_matrix(r)
    quaternion = rotation.as_quat()
    return quaternion

def get_real_pose(pos, ori, rel_pos, rel_ori):
    ori_r = quaternion_to_rotation_matrix(ori)
    real_r = get_real_r(rel_ori, ori_r)
    real_ori = r_to_quat(real_r)
    real_pos = pos + rel_pos
    return (real_pos, real_ori)