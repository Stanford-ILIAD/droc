import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
os.environ["GLOG_minloglevel"] ="2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import time
import torch
import threading
import numpy as np
import mediapipe as mp
from muse.envs.bullet_envs.control_utils import Rate
from muse.utils.transform_utils import fast_euler2quat as euler2quat, fast_quat2euler as quat2euler, \
    quat_multiply, add_euler
from muse.utils.transform_utils import *
        

# Libfranka Constants
#   > Ref: Gripper constants from: https://frankaemika.github.io/libfranka/grasp_object_8cpp-example.html
GRIPPER_SPEED, GRIPPER_FORCE, GRIPPER_MAX_WIDTH, GRIPPER_TOLERANCE = 0.08, 0.1, 0.0850, 0.01

# Joint Controller gains for recording demonstrations -- we want a compliant robot, so setting all gains to ~0.
REC_KQ_GAINS, REC_KQD_GAINS = [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]
REC_KX_GAINS, REC_KXD_GAINS = [60, 60, 60, 60, 60, 60], [60, 60, 60, 60, 60, 60]

# Hardcoded Low/High Joint Thresholds for the Franka Emika Panda Arm
LOW_JOINTS = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
HIGH_JOINTS = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

class PolymetisPandaEnv():
    def __init__(self, hz = 20, action_space= "ee-euler-delta", delta_pivot="ground-truth"):
        """
        Initialize a *physical* Franka Environment, with the given home pose, PD controller gains, and camera.

        :param params
            home: Default home position (specified in joint space - 7-DoF for Pandas)
            hz: Default policy control Hz; somewhere between 20-60 is a good range.
            use_gripper: Whether or not to initialize a gripper in addition to the robot "base"
    

        """

        # self.home = [0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, 3 * np.pi / 4.0]
        self.home = [0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0]
        self.hz = hz
        self.dt = 1. / self.hz
        self.delta_pivot = delta_pivot
        self.use_gripper =  True
        self.franka_ip =  "172.16.0.1"

        self.action_space = action_space

        self.safenet =  np.array([[-np.inf, -np.inf, 0.], [np.inf, np.inf, np.inf]])

        self.gripper, self.kq, self.kqd, self.kx, self.kxd = None, None, None, None, None


        # Pose & Robot State Trackers
        self.current_joint_pose, self.current_ee_pose, self.current_gripper_state, self.current_ee_rot = None, None, None, None
        self.initial_ee_pose, self.initial_gripper_state, self.gripper_open, self.gripper_act = None, None, True, None

        # Expected/Desired Poses (for PD Controller Deltas)
        self.expected_q, self.expected_ee_quat, self.expected_ee_euler = None, None, None
        self.desired_pose, self.use_desired_pose = {"pos": None, "ori": None}, False
        #
        # # Initialize Robot and Cartesian Impedance Controller
        # #   => Cartesian Impedance uses `HybridJointImpedanceController` so we can send `joint` or `end-effector` poses!
        # initialize the interface
        self.rate = Rate(self.hz)

    def start_controller(self) -> None:
        import torchcontrol as toco
        """Start a HybridJointImpedanceController with all 4 of the desired gains; Polymetis defaults don't set both."""
        torch_policy = toco.policies.HybridJointImpedanceControl(
            joint_pos_current=self.robot.get_joint_positions(),
            Kq=self.robot.Kq_default if self.kq is None else self.kq,
            Kqd=self.robot.Kqd_default if self.kqd is None else self.kqd,
            Kx=self.robot.Kx_default if self.kx is None else self.kx,
            Kxd=self.robot.Kxd_default if self.kxd is None else self.kxd,
            robot_model=self.robot.robot_model,
            ignore_gravity=self.robot.use_grav_comp,
        )
        self.robot.send_torch_policy(torch_policy=torch_policy, blocking=False)

    def set_controller(self) -> None:
        # Special handling *only* applicable for "kinesthetic teaching"
        self.kq, self.kqd, self.kx, self.kxd = None, None, None, None

        # Start a *Cartesian Impedance Controller* with the desired gains...
        #   Note: P/D values of "None" default to HybridJointImpedance PD defaults from Polymetis
        #         |-> These values are defined in the default launch_robot YAML (`robot_client/franka_hardware.yaml`)
        self.start_controller()


    def _reset_initialize(self, reset_gripper=True):
        from polymetis import GripperInterface, RobotInterface
        self.robot = RobotInterface(ip_address=self.franka_ip, enforce_version=False)
        if self.use_gripper:
            self.gripper = GripperInterface(ip_address=self.franka_ip)

        # WAIT a bit
        time.sleep(1.0)

        # Initialize Robot Interface and Reset to Home
        self.robot.set_home_pose(torch.Tensor(self.home))
        self.robot.go_home()
        
        # Set Robot Motion Controller (e.g., joint or cartesian impedance...)
        self.set_controller()
        # Initialize Gripper Interface & Open Gripper on each `robot_setup()`
        if self.use_gripper and reset_gripper:
            self.gripper.goto(GRIPPER_MAX_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
            # print(self.gripper.get_state().width)
            # Set Gripper State...
            self.gripper_open, self.gripper_act = True, np.array(0.0)
            self.initial_gripper_state = self.current_gripper_state = {
                "width": self.gripper.get_state().width,
                "max_width": GRIPPER_MAX_WIDTH,
                "gripper_open": self.gripper_open,
                "gripper_action": self.gripper_act,
            }

    def _reset_set_controller_and_gripper(self):

        # Initialize current joint & EE poses...
        self.current_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        self.current_ee_rot = self.current_ee_pose[3:]
        self.current_joint_pose = self.robot.get_joint_positions().numpy()

        # Set `expected` and `desired_pose` if necessary...
        self.expected_q, self.expected_ee_quat = self.current_joint_pose.copy(), self.current_ee_pose.copy()
        self.expected_ee_euler = np.concatenate([self.expected_ee_quat[:3], quat2euler(self.expected_ee_quat[3:])])
        if self.use_desired_pose:
            self.desired_pose = {"pos": self.current_ee_pose[:3], "ori": self.current_ee_rot}

    def reset(self, reset_gripper=True):
        
        self._reset_initialize(reset_gripper=reset_gripper)
        self._reset_set_controller_and_gripper()
        self.initial_ee_quat = self.ee_orientation.copy()
        # Return initial observation
        return self.get_obs()

    
    def compute_reward(self, obs):
        # override this in sub-classes
        obs["reward"] = np.array([0.])
        return obs

    def get_done(self):
        return np.array([False])

    def get_obs(self):
        new_joint_pose = self.robot.get_joint_positions().numpy()
        new_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        new_ee_rot = new_ee_pose[3:]

        curr_obs = {
            "q":new_joint_pose,
            "qdot":self.robot.get_joint_velocities().numpy(),
            "ee_pose":new_ee_pose,
            "ee_position":new_ee_pose[:3],
            "ee_orientation":new_ee_rot,
            "ee_orientation_eul":quat2euler(new_ee_rot),
        }
        self.current_joint_pose, self.current_ee_pose, self.current_ee_rot = new_joint_pose, new_ee_pose, new_ee_rot

        if self.use_gripper:
            new_gripper_state = self.gripper.get_state()

            curr_obs["gripper_width"]=np.array([new_gripper_state.width]),
            curr_obs["gripper_pos"]=np.array([1 - new_gripper_state.width / GRIPPER_MAX_WIDTH]),  # 1 for closed
            curr_obs["gripper_max_width"]=np.array([GRIPPER_MAX_WIDTH]),
            curr_obs["gripper_open"]=np.array([self.gripper_open]),
            curr_obs["gripper_action"]=self.gripper_act[None],
            
            self.current_gripper_state = {
                "width": self.gripper.get_state().width,
                "max_width": GRIPPER_MAX_WIDTH,
                "gripper_open": self.gripper_open,
                "gripper_action": self.gripper_act,
            }

      
        curr_obs = self.compute_reward(curr_obs)

        return curr_obs

    def step(self, action):
        #action = to_numpy(act.action, check=True).reshape(-1)
        open_gripper = action[-1] <= 0.5  # 0 for open, 1 for closed.
        """Run an environment step, where `delta` specifies if we are sending absolute poses or finite differences."""
        if action is not None:
            if self.action_space == "joint":
                self.robot.update_desired_joint_positions(torch.from_numpy(action).float())
                self.expected_ee_euler = np.concatenate(
                    [self.current_ee_pose[:3], quat2euler(self.current_ee_pose[3:])])

            elif self.action_space == "joint-delta":
                if self.delta_pivot == "ground-truth":
                    next_q = self.current_joint_pose + action
                elif self.delta_pivot == "expected":
                    next_q = self.expected_q = self.expected_q + action
                else:
                    raise ValueError(f"Delta Pivot `{self.delta_pivot}` not supported!")

                # Act!
                self.robot.update_desired_joint_positions(torch.from_numpy(next_q).float())
                self.expected_ee_euler = np.concatenate(
                    [self.current_ee_pose[:3], quat2euler(self.current_ee_pose[3:])])

            elif self.action_space == "ee-quat":
                # Compute quaternion from euler...
                desired_pos, desired_quat = action[:3], action[3:7]

                # Send to controller =>> Both position & orientation control!
                self.robot.update_desired_ee_pose(
                    position=torch.from_numpy(desired_pos).float(),
                    orientation=torch.from_numpy(desired_quat).float(),
                )
                self.expected_q = self.current_joint_pose

            elif self.action_space == "ee-euler":
                # Compute quaternion from euler...
                desired_pos, desired_quat = action[:3], euler2quat(action[3:6])

                # Send to controller =>> Both position & orientation control!
                self.robot.update_desired_ee_pose(
                    position=torch.from_numpy(desired_pos).float(),
                    orientation=torch.from_numpy(desired_quat).float(),
                )
                self.expected_q = self.current_joint_pose

            elif self.action_space == "ee-euler-delta":
                if self.delta_pivot == "ground-truth":
                    next_pos = self.current_ee_pose[:3] + action[:3]
                    next_quat = quat_multiply(euler2quat(action[3:6]), self.current_ee_rot)
                elif self.delta_pivot == "expected":
                    next_pos = self.expected_ee_euler[:3] = self.expected_ee_euler[:3] + action[:3]
                    self.expected_ee_euler[3:] = add_euler(action[3:6], self.expected_ee_euler[3:])
                    next_quat = euler2quat(self.expected_ee_euler[3:])
                elif self.delta_pivot ==  "eval":
                    next_pos = self.expected_ee_euler[:3] = self.current_ee_pose[:3] + 0.25*(self.expected_ee_euler[:3]-self.current_ee_pose[:3]) + action[:3]
                    next_quat = quat_multiply(euler2quat(action[3:6]), self.current_ee_rot)
                else:
                    raise ValueError(f"Delta Pivot `{self.delta_pivot}` not supported!")

                # Send to controller =>> Both position & orientation control!
                self.robot.update_desired_ee_pose(
                    position=torch.from_numpy(next_pos).float(),
                    orientation=torch.from_numpy(next_quat).float(),
                )
                self.expected_q = self.current_joint_pose

            else:
                raise NotImplementedError(f"Support for Action Space `{self.action_space}` not yet implemented!")

        # Discrete Grasping (Open/Close)
        if open_gripper is not None and (self.gripper_open ^ open_gripper):
            # True --> Open Gripper, otherwise --> Close Gripper
            self.gripper_open = open_gripper
            if open_gripper:
                self.gripper.goto(GRIPPER_MAX_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE, blocking=True)
            else:
                self.gripper.grasp(speed=GRIPPER_SPEED, force=GRIPPER_FORCE, blocking=True)

        # Sleep according to control frequency
        self.rate.sleep()

        # Return observation, Gym default signature...
        return self.get_obs(), self.get_done()

    @property
    def ee_position(self) -> np.ndarray:
        """Return current EE position --> 3D x/y/z."""
        return self.current_ee_pose[:3] if not self.use_desired_pose else self.desired_pose["pos"]

    @property
    def ee_orientation(self) -> np.ndarray:
        """Return current EE orientation --> quaternion [i, j, k, w]."""
        return self.current_ee_rot if not self.use_desired_pose else self.desired_pose["ori"]

    @property
    def ground_truth_ee_pose(self) -> np.ndarray:
        return np.concatenate([ee.numpy() for ee in self.robot.get_ee_pose()])

    @property
    def ground_truth_joint_state(self) -> np.ndarray:
        return self.robot.get_joint_positions().numpy()

    def toggle_grasp(self, open_gripper=True):
        # Discrete Grasping (Open/Close)
        if open_gripper is not None and (self.gripper_open ^ open_gripper):
            # True --> Open Gripper, otherwise --> Close Gripper
            self.gripper_open = open_gripper
            if open_gripper:
                self.gripper.goto(GRIPPER_MAX_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE, blocking=True)
                #self.gripper.grasp(speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
            else:
                self.gripper.grasp(speed=GRIPPER_SPEED, force=GRIPPER_FORCE, blocking=True)
        self.rate.sleep()

    def set_gripper_width(self, width):
        self.gripper.goto(width=width, speed=0.05, force=0.1)

    def open_gripper(self):
        self.gripper.goto(GRIPPER_MAX_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
        self.gripper_open = True
        return self.get_obs()

    def close_gripper(self):
        self.gripper.grasp(speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
        self.gripper_open = False
        return self.get_obs()

    def close(self) -> None:
        # Terminate Policy
        self.robot.terminate_current_policy()

        # Garbage collection & sleep just in case...
        del self.robot
        self.robot, self.gripper = None, None
        time.sleep(1)


if __name__ == '__main__':
    robot = PolymetisPandaEnv()
    robot.reset()
    
    robot.toggle_grasp(open_gripper=False)
    time.sleep(0.5)
    robot.toggle_grasp(open_gripper=True)
    time.sleep(0.5)

    robot.close()
