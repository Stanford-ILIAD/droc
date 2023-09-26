import numpy as np
from utils.exception_utils import InterruptedByHuman

class DummyPolicy:
    def __init__(self):
        pass

    def close_gripper(self, check_grasp=True):
        return True
        
    def open_gripper(self, width=1.0):
        pass
  
    def rotate_gripper(self, degrees, axis):
        return np.array((1.,0.,0.,0.))


    def move_to_pos(self, ee_pos, tar_quat):
        a = input('Do you want to raise exception?')
        if a == 'y':
            raise InterruptedByHuman('')
        pass

    def rotate_around_gripper_z_axis(self, angle, quat=None):
        return np.array((1.,0.,0.,0.))

    def tilt_leftright(self, degrees):
        return np.array((1.,0.,0.,0.))

    def tilt_updown(self, degrees=None):
        return np.array((1.,0.,0.,0.))

    def align_z_axis_with_vector(self, z_axis, finger_plane='vertical'):
        return np.array((1.,0.,0.,0.))

    def reset(self):
        pass

    def robot_fingertip_pos_to_ee(self, fingertip_pos, ee_quat):
        return np.array((1.,0.,0.))

    def ee_pos_to_fingertip(self, ee_pos, ee_quat):
        return np.array((1.,0.,0.))

    def get_horizontal_ori(self):
        return np.array((1.,0.,0.,0.))

    def get_vertical_ori(self):
        return np.array((1.,0.,0.,0.))