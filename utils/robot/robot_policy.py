import time
import numpy as np
from utils.exception_utils import InterruptedByHuman, RobotError, GraspError
from utils.robot.panda_env import PandaEnv
from scipy.spatial.transform import Rotation
from utils.transformation_utils import extract_z_axis, pose_to_mat, quat_to_euler, add_euler, euler_to_quat, quat_to_mat, mat_to_quat, mat_to_euler


GRIPPER_SPEED, GRIPPER_FORCE, GRIPPER_MAX_WIDTH, GRIPPER_TOLERANCE = 0.1, 40, 0.08570, 0.01

def calculate_frame_quaternion(g_x, g_z):
    base_x = np.array([1., 0., 0.])  # Replace with the X-axis vector of the base frame
    base_z = np.array([0., 0., 1.])  # Replace with the Z-axis vector of the base frame
    # Step 1: Calculate Y-axis vector of frame G in the base frame
    base_y = np.cross(base_z, base_x)
    base_y /= np.linalg.norm(base_y)
    # Step 2: Create rotation matrices for the base and frame G
    R_base = np.column_stack((base_x, base_y, base_z))
    R_g = np.column_stack((g_x, np.cross(g_z, g_x), g_z))      
    # Step 3: Compute rotation matrix to transform frame G's axes to the base frame axes
    R_relative_to_base = np.dot(R_base.T, R_g)
    # Step 4: Convert the rotation matrix to a quaternion using scipy's Rotation class
    r = Rotation.from_matrix(R_relative_to_base)
    frame_quaternion = r.as_quat()
    return frame_quaternion

class KptPrimitivePolicy:
    def __init__(self):
        self.robot_env = PandaEnv()
        self.robot_env.reset()

    def close_gripper(self, check_grasp=True):
        self.robot_env.gripper.grasp(speed=GRIPPER_SPEED, force=GRIPPER_FORCE, blocking=True)
        time.sleep(1.8)
        obj_in_gripper = False
        if self.robot_env.gripper.get_state().width > 0.005:
            obj_in_gripper = True
        if obj_in_gripper == False:
            if check_grasp:
                raise GraspError('Grasp Failure')
        return obj_in_gripper
        
    def open_gripper(self, width=1.0):
        self.robot_env.gripper.goto(width*GRIPPER_MAX_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
  
    def rotate_gripper(self, degrees, axis):
        if axis == 'z':
            tar_quat = self.rotate_around_gripper_z_axis(degrees)
        elif axis == 'x':
            tar_quat = self.tilt_updown(degrees)
        elif axis == 'y':
            tar_quat = self.tilt_leftright(degrees)
        return tar_quat


    def move_to_pos(self, ee_pos, tar_quat):
    
        assert len(tar_quat) == 4
        if ee_pos[2] > 0.5:
            ee_pos[2] = 0.5
        if ee_pos[2] < 0.115:
            ee_pos[2] = 0.115
        # ee_pos = self.robot_fingertip_pos_to_ee(fingertip_pos, tar_quat)
        ret_val = self.robot_env.robot.move_to_ee_pose(position=ee_pos, orientation=tar_quat)
        if ret_val == 1:
            raise InterruptedByHuman('Interrupted by human.')
        elif ret_val == 2:
            raise RobotError('IK did not converge.')

    def rotate_around_gripper_z_axis(self, angle, quat=None):
        theta_rad = np.radians(angle)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        r_gd = np.array([[cos_theta, -sin_theta, 0],
                  [sin_theta, cos_theta, 0],
                  [0, 0, 1]])
        pos = self.robot_env.robot.get_ee_pose()
        if quat is None:
            t_bg = pose_to_mat((pos[0].cpu().numpy(), pos[1].cpu().numpy()))
        else:
            t_bg = pose_to_mat((pos[0].cpu().numpy(), np.array(quat)))
        r_bg = t_bg[:3, :3]
        r_bd = r_bg @ r_gd
        r_bd = Rotation.from_matrix(r_bd)
        q_b = r_bd.as_quat()
        return q_b

    def tilt_leftright(self, degrees):
        current_euler = quat_to_euler(self.robot_env.robot.get_ee_pose()[1].numpy())
        angle = np.radians(degrees)
        delta_euler = np.array((0., 0., angle))   # tilt_leftright rotate around absolute z-axis
        new_euler = add_euler(delta_euler, current_euler)
        tar_quat = euler_to_quat(new_euler)
        return tar_quat

    def tilt_updown(self, degrees=None):
        if degrees is None:
            current_z_axis = extract_z_axis(self.robot_env.robot.get_ee_pose()[1].numpy())
            old_x = current_z_axis[0]
            old_y = current_z_axis[1]
            if np.abs(old_y) >= 10e-3:
                c = old_x/old_y
                new_y = 1/np.sqrt(c**2+1) if current_z_axis[1]>0 else -1/np.sqrt(c**2+1)
                new_x = c*new_y
                new_z_axis = np.array((new_x, new_y, 0))
                new_x_axis = self.g_x
            else:
                new_z_axis = np.array((1., 0., 0.))
                new_x_axis = self.g_x
            frame_quaternion = calculate_frame_quaternion(new_x_axis, new_z_axis)
            tar_quat = self.rotate_around_gripper_z_axis(45, frame_quaternion)
        else:
            g_z = extract_z_axis(self.robot_env.robot.get_ee_pose()[1].numpy())
            g_z[2] = 0
            g_x = np.array((g_z[1],-g_z[0],0))
            g_y = np.cross(g_z, g_x)
            g_x = g_x/np.linalg.norm(g_x)
            g_y = g_y/np.linalg.norm(g_y)
            g_z = g_z/np.linalg.norm(g_z)
            r_b = np.stack((g_x,g_y,g_z),axis=1)
            r_sb = r_b
            r_bs = r_sb.T
            mat_g_in_s = quat_to_mat(self.robot_env.robot.get_ee_pose()[1].numpy())
            mat_g_in_b = r_bs.dot(mat_g_in_s)
            euler_g_in_b = mat_to_euler(mat_g_in_b)
            angle = np.radians(degrees)
            delta_euler = np.array((angle, 0., 0.))   # tilt_updown rotate around gripper x-axis
            new_euler = add_euler(delta_euler, euler_g_in_b)
            new_quat_g_in_b = euler_to_quat(new_euler)
            new_mat_g_in_b = quat_to_mat(new_quat_g_in_b)
            new_mat_g_in_s = r_sb.dot(new_mat_g_in_b)
            tar_quat = mat_to_quat(new_mat_g_in_s)
        return tar_quat

    def align_z_axis_with_vector(self, z_axis, finger_plane='vertical'):
        g_z = z_axis
        self.g_z = g_z
        if finger_plane == 'horizontal':
            g_x = np.array([0.,0.,1.])
            self.g_x = g_x
            self.g_y = np.cross(g_z, g_x)
        elif finger_plane == 'vertical':
            if g_z[0]*g_z[1] >=0:
                g_y = np.array([0.,0.,1.])
            else:
                g_y = np.array([0.,0.,-1.])
            g_x = np.cross(g_z, g_y)
            self.g_x = g_x
            self.g_y = g_y
        frame_quaternion = calculate_frame_quaternion(g_x, g_z)
        frame_quaternion = self.rotate_around_gripper_z_axis(45, frame_quaternion)
        return frame_quaternion

    def reset(self,reset_gripper=True):
        self.robot_env.reset(reset_gripper=reset_gripper)

    def robot_fingertip_pos_to_ee(self, fingertip_pos, ee_quat):
        HOME_QUAT = np.array([ 0.9201814 , -0.39136365,  0.00602445,  0.00802529])
        FINGERTIP_OFFSET = np.array([0,0,-0.095])
        home_euler = Rotation.from_quat(HOME_QUAT).as_euler('zyx', degrees=True)
        ee_euler = Rotation.from_quat(ee_quat).as_euler('zyx', degrees=True)
        offset_euler = ee_euler - home_euler
        fingertip_offset_euler = offset_euler * [1,-1,1]
        fingertip_transf = Rotation.from_euler('zyx', fingertip_offset_euler, degrees=True)
        fingertip_offset = fingertip_transf.as_matrix() @ FINGERTIP_OFFSET
        fingertip_offset[2] -= FINGERTIP_OFFSET[2]
        ee_pos = fingertip_pos - fingertip_offset
        return ee_pos

    def ee_pos_to_fingertip(self, ee_pos, ee_quat):
        current_z_aixs = extract_z_axis(ee_quat)
        tip_pos = 0.095 * current_z_aixs + ee_pos
        return tip_pos

    def get_horizontal_ori(self):
        current_z_aixs = extract_z_axis(self.robot_env.robot.get_ee_pose()[1].numpy())
        if np.abs(current_z_aixs[2]) > 0.9:
            target_z_aixs = np.array((1.,0.,0.))
        else:
            current_z_aixs[2] = 0.
            current_z_aixs = -current_z_aixs if current_z_aixs[0]<0 else current_z_aixs
            target_z_aixs = current_z_aixs/np.linalg.norm(current_z_aixs)
        quat_tmp = self.align_z_axis_with_vector(target_z_aixs)
        quat = self.rotate_around_gripper_z_axis(-90, quat_tmp)
        return quat

    def get_vertical_ori(self):
        target_z_aixs = np.array((0.,0.,-1.))
        return self.align_z_axis_with_vector(target_z_aixs)
