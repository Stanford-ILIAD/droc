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

def pose_to_mat(pose):
    homo_pose_mat = np.zeros((4, 4), dtype=np.float32)
    homo_pose_mat[:3, :3] = quat2mat(pose[1])
    homo_pose_mat[:3, 3] = np.array(pose[0], dtype=np.float32)
    homo_pose_mat[3, 3] = 1.
    return homo_pose_mat
    
def quat_to_euler(quat):
    assert quat.shape[-1] == 4
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    ysqr = y * y
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)
    return np.stack([X, Y, Z], axis=-1)
    
def add_euler(delta, source, degrees=False):
    delta_rot = R.from_euler('xyz', delta, degrees=degrees)
    source_rot = R.from_euler('xyz', source, degrees=degrees)
    new_rot = delta_rot * source_rot
    return new_rot.as_euler('xyz', degrees=degrees)

def euler_to_quat(euler):
    roll, pitch, yaw = euler
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    
    return [qx, qy, qz, qw]

def quat_to_mat(euler):
    return R.from_euler("xyz", euler).as_quat()

def mat_to_quat(mat):
    return R.from_matrix(mat).as_quat()

def mat_to_euler(mat):
    return T.rmat_to_euler(mat, 'XYZ')

def quat_multiply(quat0, quat1):
    x0, y0, z0, w0 = np.split(quat0, 4, axis=-1)  # (..., 1) for each
    x1, y1, z1, w1 = np.split(quat1, 4, axis=-1)
    return np.concatenate(
        [
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,  # (..., 1)
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ],
        axis=-1
    )