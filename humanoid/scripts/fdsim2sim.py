"""fdrobot sim2sim with ros2 topics."""

import math
import os
print(os.getenv('PYTHONPATH'))
# import select
import sys  # 导入系统模块
import termios  # 导入termios模块,用于保存和恢复终端属性
import time
# import tty  # 导入tty模块,用于设置终端模式
from collections import deque
# from functools import partial

workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(workspace_dir)  # append workspace directory to PYTHONPATH

import mujoco
import mujoco.viewer
#import mujoco_py
# import mujoco_viewer

# from tqdm import tqdm

import numpy as np

from pynput import keyboard

# import rclpy
# import rclpy.time  # 导入select模块,用于实现输入监听
# from rclpy.node import Node
# from control_msgs.msg import JointTrajectoryControllerState
# from trajectory_msgs.msg import JointTrajectoryPoint as JTP
# from sensor_msgs.msg import Imu

from scipy.spatial.transform import Rotation as R

import torch

from humanoid.tools.timer import Timer
import humanoid.config.sim2sim_config as sim2simcfg
import humanoid.tools.common as tool_util
import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import Float32


settings = termios.tcgetattr(sys.stdin)  # 获取标准输入终端属性
x_time = 0


def time_decorator(func):
    """Print elapsed time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'{func.__name__} time-consuming:{elapsed_time:.2f}s')
    return wrapper


class cmd:
    """Robot command."""
    vx = 0.0
    vy = 0.0
    dyaw = 0.0


def on_key_press(key):
    """Get key press."""
    try:
        # print(key)
        ch = key.char.lower()
        if ch not in 'wsadqek':
            return
        keyboard_control(ch)
    except AttributeError:
        pass


def keyboard_control(key):
    """."""
    if key == 'w':
        cmd.vx = cmd.vx + 0.3
        if cmd.vx > 1.5:
            cmd.vx = 1.5
    if key == 's':
        cmd.vx = cmd.vx - 0.3
        if cmd.vx < -0.6:
            cmd.vx = -0.6
    if key == 'q':
        cmd.vy = cmd.vy + 0.1
        if cmd.vy > 0.6:
            cmd.vy = 0.6
    if key == 'e':
        cmd.vy = cmd.vy - 0.1
        if cmd.vy < -0.6:
            cmd.vy = -0.6
    if key == 'a':
        cmd.dyaw = cmd.dyaw - 0.2
        if cmd.dyaw < -0.6:
            cmd.dyaw = -0.6
    if key == 'd':
        cmd.dyaw = cmd.dyaw + 0.2
        if cmd.dyaw > 0.6:
            cmd.dyaw = 0.6

    if key == 'k':
        cmd.vx = 0
        cmd.vy = 0
        cmd.dyaw = 0


def quaternion_to_euler_array(quat):
    """Quaternion to euler array."""
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])


def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵."""
    q /= np.linalg.norm(q)

    x, y, z, w = q
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
    ])
    return R


# class HumaRobotSimRos(Node):
class HumaRobotSimRos():
    """Sim with Ros message sent, for fdrobot HumaRobot T1."""
    def __init__(self, node_name) -> None:
        """."""
        # super().__init__(node_name)
        # self.sim_state_publisher = self.create_publisher(JointTrajectoryControllerState, 'sim_ros/joint_states', 10)

        self.dof = 12
        # self.sim_state_publish_msg = JointTrajectoryControllerState()
        # self.sim_state_publish_msg.joint_names = [
        #     'joint_L_hipY', 'joint_L_hipR', 'joint_L_hipP', 'joint_L_knee', 'joint_L_ankleR', 'joint_L_ankleP',
        #     'joint_R_hipY', 'joint_R_hipR', 'joint_R_hipP', 'joint_R_knee', 'joint_R_ankleR', 'joint_R_ankleP'
        # ]
        # self.sim_imu_publisher = self.create_publisher(Imu, 'sim_ros/imu', 10)

        # state
        self.rl_last_time = None  # time.time()
        self._gait_time = 0.0  # current gait time [0, step_sycle_time]

        # config
        self.step_sycle_time = 0.64
        self.action_delay = 0.85

    def get_obs(self, data):
        """Extracts an observation from the mujoco data structure."""
        q = data.qpos.astype(np.double)
        dq = data.qvel.astype(np.double)
        quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
        r = R.from_quat(quat)
        v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
        omega = data.sensor('angular-velocity').data.astype(np.double)
        gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
        lin_acc = data.sensor('linear-acceleration').data.astype(np.double)
        torque_foot = np.zeros([4])
        torque_foot[0] = data.sensor('LeftFootForceSensor_tsensor').data.astype(np.double)[0]
        torque_foot[1] = data.sensor('RightFootForceSensor_tsensor').data.astype(np.double)[0]
        torque_foot[2] = data.sensor('LeftFootForceSensor_tsensor').data.astype(np.double)[1]
        torque_foot[3] = data.sensor('RightFootForceSensor_tsensor').data.astype(np.double)[1]
        force_foot = np.zeros([2])
        force_foot[0] = -data.sensor('LeftFootForceSensor_fsensor').data.astype(np.double)[2]
        force_foot[1] = -data.sensor('RightFootForceSensor_fsensor').data.astype(np.double)[2]
        

        # pos = data.sensor('position').data.astype(np.double)
        # posr = data.sensor('positionr').data.astype(np.double)
        # posl = data.sensor('positionl').data.astype(np.double)
        # print("机器人位置：", pos)
        # print("机器人位置r:", posr)
        # print("机器人位置l:", posl)
        src_force_foot = force_foot.copy()

        force_foot[force_foot < 0] = 0
        force_foot = force_foot > 10
        # print(force_foot)
        lin_vel = data.sensor('linear-velocity').data.astype(np.double)

        return (q, dq, quat, v, omega, gvec, lin_acc, torque_foot, lin_vel, force_foot, src_force_foot)

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        """Calculates torques from position commands."""
        return (target_q - q) * kp + (target_dq - dq) * kd

    def get_sin_gait(self, phase):
        """产生平滑过渡轨迹, 替换 torch.sin(2 * torch.pi * phase)."""
        sin_pos = (math.sin(phase * np.pi * 4. - 0.5 * np.pi) + 1) / 2.
        if (phase % 1) > 0.5:
            sin_pos = - sin_pos
        return sin_pos

    @time_decorator
    def run_mujoco(self, policy, param, cfg : sim2simcfg.Sim2simCfg):
        """Run the Mujoco simulation using the provided policy and configuration.

        Args:
            policy: The policy used for controlling the simulation.
            cfg: The configuration object containing simulation settings.

        Returns:
            None
        """
        self.param = param
        self.policy = policy

        model = mujoco.MjModel.from_xml_path(cfg.sim_config.mjcf_path)
        print(cfg.sim_config.mjcf_path)
        model.opt.timestep = cfg.sim_config.dt
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        n_joints = model.njnt
        for i in range(n_joints):
            # 获取关节名称
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            # 打印关节名称
            print(f'关节 {i}: {joint_name}\r')

        # viewer = mujoco_viewer.MujocoViewer(model, data)
        viewer = mujoco.viewer.launch_passive(model, data)
        target_q = np.zeros((param['num_actions']), dtype=np.double)
        self.action = np.zeros((param['num_actions']), dtype=np.double)
        self.last_action = np.zeros((param['num_actions']), dtype=np.double)

        # init obs
        self.obs_torques = np.zeros((param['num_actions']), dtype=np.double)
        self.obs_torques_mark = np.zeros((param['num_actions']), dtype=np.double)
        self.obs_torques_mark[4:6] = 1.0
        self.obs_torques_mark[10:12] = 1.0  # observate joints: 4-6 10-12 only

        self.old_lin_vel = np.zeros([3], dtype=np.double)  # waist linear vel
        self.waist_lin_acc_from_vel = np.zeros([3], dtype=np.double)  # waist linear acc

        self.hist_obs = deque()
        for _ in range(param['frame_stack']):
            self.hist_obs.append(np.zeros([1, param['num_single_obs']], dtype=np.double))

        count_lowlevel = 0

        pd_kps = np.array(cfg.robot_config.kps, dtype=np.double)
        pd_kds = np.array(cfg.robot_config.kds, dtype=np.double)
        tau_limit = np.array(cfg.robot_config.tau_limit, dtype=np.double)

        ref_time = time.time()
        # for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc='Simulating...'):
        with Timer(f'[INFO]: Time taken for {cfg.sim_config.sim_duration:.2f}s simulation\r'):
            for _ in range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)):
                # Obtain an observation
                q, dq, quat, v, omega, gvec, lin_acc, torque_foot, lin_vel, force_foot, src_frc_foot = self.get_obs(data)
                q = q[-param['num_actions']:]
                dq = dq[-param['num_actions']:]
                rBody = quaternion_to_rotation_matrix(quat)

                # 1000hz -> 100hz
                if count_lowlevel % cfg.sim_config.decimation == 0:
                    target_q = self.rl_policy_forward(param=param, policy=policy, q=q, dq=dq, waist_orient_quat=quat,
                                                      waist_omega=omega, torque_foot=torque_foot, force_foot=force_foot,
                                                      lin_vel=lin_vel)
                # 低级控制
                target_dq = np.zeros((param['num_actions']), dtype=np.double)
                # Generate PD control
                tau = self.pd_control(target_q, q, pd_kps, target_dq, dq, pd_kds)  # Calc torques
                tau = np.clip(tau, -tau_limit, tau_limit)  # Clamp torques

                data.ctrl = tau

                # 跌到重置
                if rBody[2, 2] < 0.5:
                    mujoco.mj_resetData(model, data)

                mujoco.mj_step(model, data)
                ref_time += model.opt.timestep

                # if count_lowlevel % 10 == 0:
                #     # 状态发布
                #     now_time = time.time()
                #     self.publish_state(
                #         now_time,
                #         current_state=JTP(positions=q, velocities=dq, effort=self.sim_effort_state(torque_foot)),
                #         disired_state=JTP(positions=target_q, velocities=np.zeros(12), effort=tau),
                #         error_state=self.sim_error_state(JTP(), src_frc_foot))

                #     self.publish_imu(now_time, quat, omega, self.waist_lin_acc_from_vel)
            
                # y = self.waist_lin_acc_from_vel[0]
                # pub.publish(y)
                # global x_time 
                # x_time = x_time + 1
                # plt.plot(x_time,y)
                # plt.xlabel('Time')
                # plt.ylabel('Acc')
                # plt.grid(True)
                # plt.show(block=False)
                # plt.pause(0.01)

                

                if count_lowlevel % 10 == 0:
                    # viewer.render()
                    viewer.sync()

                if count_lowlevel % 10 == 0:  # 100HZ time sync
                    cur_time = time.time()
                    delta_time_s = ref_time - cur_time
                    if delta_time_s > 0:
                        time.sleep(delta_time_s)

                count_lowlevel += 1
            pass
        viewer.close()

    def rl_policy_forward(self, param, policy, q, dq, waist_orient_quat, waist_omega, torque_foot, force_foot, lin_vel):
        """."""
        obs = np.zeros([1, param['num_single_obs']], dtype=np.float32)
        # 加速度观测
        self.waist_lin_acc_from_vel = (self.old_lin_vel - lin_vel) / 0.01
        self.old_lin_vel = lin_vel.copy()
        # 扭矩观测
        self.obs_torques[4] = torque_foot[0]
        self.obs_torques[5] = torque_foot[2]
        self.obs_torques[10] = torque_foot[1]
        self.obs_torques[11] = torque_foot[3]
        # 欧拉角
        eu_ang = quaternion_to_euler_array(waist_orient_quat)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi
        clk_a, clk_b, phase = self.gen_rl_clock()
        obs[0, 0] = clk_a
        obs[0, 1] = clk_b
        obs[0, 2] = cmd.vx * param['obs_scales_lin_vel']
        obs[0, 3] = cmd.vy * param['obs_scales_lin_vel']
        obs[0, 4] = cmd.dyaw * param['obs_scales_ang_vel']
        obs[0, param['obs_index_q'] : param['obs_index_q'] + len(q)] = q * param['obs_scales_dof_pos']
        obs[0, param['obs_index_qd'] : param['obs_index_qd'] + len(dq)] = dq * param['obs_scales_dof_vel']
        obs[0, param['obs_index_action'] : param['obs_index_action'] + len(self.action)] = self.action
        obs[0, param['obs_index_imu_waist_angvel'] : param['obs_index_imu_waist_angvel'] + len(waist_omega)] =\
            waist_omega * param['obs_scales_imu_waist_angvel']
        obs[0, param['obs_index_imu_waist_orientation'] : param['obs_index_imu_waist_orientation'] + len(eu_ang)] =\
            eu_ang * param['obs_scales_imu_waist_orientation']
        if param['has_obs_torque']:
            obs[0, param['obs_index_torque'] : param['obs_index_torque'] + len(self.obs_torques)] =\
                self.obs_torques * self.obs_torques_mark * param['obs_scales_torque'] * 0.1
        if param['has_obs_imu_waist_linearacc']:
            obs[0, param['obs_index_imu_waist_linearacc'] : param['obs_index_imu_waist_linearacc'] + len(self.waist_lin_acc_from_vel)] = \
                self.waist_lin_acc_from_vel * param['obs_scales_imu_waist_linearacc']
        if param['has_obs_forcesensor']:
            obs[0, param['obs_index_forcesensor'] : param['obs_index_forcesensor'] + len(force_foot)] = \
                force_foot * param['obs_scales_force']
        obs = np.clip(obs, -param['clip_observations'], param['clip_observations'])
        self.hist_obs.append(obs)
        self.hist_obs.popleft()

        policy_input = np.zeros([1, param['num_observations']], dtype=np.float32)
        for i in range(param['frame_stack']):
            policy_input[0, i * param['num_single_obs'] : (i + 1) * param['num_single_obs']] = self.hist_obs[i][0, :]

        self.action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()

        self.action = np.clip(self.action, -param['clip_actions'], param['clip_actions'])
        self.action = self.action_delay * self.last_action + (1 - self.action_delay) * self.action
        self.last_action = self.action.copy()
        target_q = self.action * param['action_scale']
        # target_q = np.zeros_like(target_q)
        # target_q[2] = 0.25
        # target_q[3] = 0.5
        # target_q[5] = 0.25
        return target_q

    def gen_rl_clock(self):
        """."""
        cur_time = time.time()
        if self.rl_last_time is None:
            period = 0.0
        else:
            period = cur_time - self.rl_last_time
        self.rl_last_time = cur_time

        self._gait_time += period
        if self._gait_time > self.step_sycle_time:
            self._gait_time -= self.step_sycle_time
        phase = self._gait_time / self.step_sycle_time
        if param['ref_cycle_type'] == 1:
            clock_a = math.sin(2 * math.pi * phase)
            clock_b = math.sin(2 * math.pi * phase)

            if clock_a > 0:
                clock_a = 0
            else:
                clock_a = -clock_a

            if clock_a < 0.1:
                clock_a = 0

            if clock_b < 0:
                clock_b = 0

            if clock_b < 0.1:
                clock_b = 0
        else:
            clock_a = math.sin(2 * math.pi * phase)
            clock_b = math.cos(2 * math.pi * phase)
        if cmd.vx == 0. and cmd.vy == 0. and cmd.dyaw == 0.:
            clock_a = 0
            clock_b = 0
        return clock_a, clock_b, phase

    # def publish_imu(self, now_time, waist_orient_quat, ang_vel, lin_acc):
    #     """Publis Imu Data."""
    #     sec = int(now_time)
    #     nanoseconds = int((now_time - sec) * 1e9)
    #     eu_ang = quaternion_to_euler_array(waist_orient_quat)
    #     imu_data = Imu()
    #     imu_data.header.stamp.sec = sec
    #     imu_data.header.stamp.nanosec = nanoseconds
    #     imu_data.orientation.x = eu_ang[0]
    #     imu_data.orientation.y = eu_ang[1]
    #     imu_data.orientation.z = eu_ang[2]
    #     imu_data.angular_velocity.x = ang_vel[0]
    #     imu_data.angular_velocity.y = ang_vel[1]
    #     imu_data.angular_velocity.z = ang_vel[2]
    #     imu_data.linear_acceleration.x = lin_acc[0]
    #     imu_data.linear_acceleration.y = lin_acc[1]
    #     imu_data.linear_acceleration.z = lin_acc[2]
    #     self.sim_imu_publisher.publish(imu_data)

    # def publish_state(self, now_time, current_state, disired_state, error_state):
    #     """Publish Sim Robot JointStates."""
    #     sec = int(now_time)
    #     nanoseconds = int((now_time - sec) * 1e9)
    #     self.sim_state_publish_msg.header.stamp.sec = sec
    #     self.sim_state_publish_msg.header.stamp.nanosec = nanoseconds

    #     self.sim_state_publish_msg.reference = disired_state
    #     self.sim_state_publish_msg.feedback = current_state
    #     self.sim_state_publish_msg.error = error_state
    #     self.sim_state_publisher.publish(self.sim_state_publish_msg)

    # def sim_effort_state(self, torque_foot):
    #     """Extract effort state."""
    #     effort = np.zeros(self.dof)
    #     effort[4] = torque_foot[0]
    #     effort[5] = torque_foot[2]
    #     effort[10] = torque_foot[1]
    #     effort[11] = torque_foot[3]
    #     return effort

    # def sim_error_state(self, error_state: JTP, src_frc_foot):
    #     """Temporary: used to transmit sensors's status."""
    #     error_state.effort = np.zeros(self.dof)
    #     error_state.effort[5] = src_frc_foot[0]
    #     error_state.effort[11] = src_frc_foot[1]
    #     return error_state


if __name__ == '__main__':
    # rclpy.init(args=None)
    import argparse
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--cfg', type=str, required=False, help='Run to load from.')
    parser.add_argument('--load_model', type=str, required=False, help='model to load.')
    parser.add_argument('--model_param', type=str, required=False, help='param for model')
    parser.add_argument('--experiment_name', type=str, required=False, help='param for model')
    parser.add_argument('--use_local_cfg', action='store_true', required=False, help='use local source code cfg')

    args = parser.parse_args()

    if args.experiment_name is not None:
        experiment_name = args.experiment_name
    else:
        experiment_name = 'FdBot_ppo'

    # use config file default(load_cfg = True)
    if args.use_local_cfg:
        load_cfg = False
    else:
        load_cfg = True

    cfg = sim2simcfg.Sim2simCfg()
    cfg.sim_config.sim_duration = 100000.0
    cfg.sim_config.mjcf_path = cfg.sim_config.mjcf_path.format(workspace_dir=workspace_dir)
    cfg.sim_config.model_path = cfg.sim_config.model_path.format(
        workspace_dir=workspace_dir, experiment_name=experiment_name)
    print(cfg.sim_config.mjcf_path)
    
    cfg.sim_config.param_path = cfg.sim_config.param_path.format(
        workspace_dir=workspace_dir, experiment_name=experiment_name)

    # load config if needed
    if load_cfg:
        if args.cfg is not None:
            cfgfile = args.cfg.format(workspace_dir=workspace_dir)
        else:
            cfgfile = f'{workspace_dir}/logs/{experiment_name}/exported/policies/{tool_util.SIM2SIMCFG_FILENAME}'
        tool_util.update_class_from_yaml_file(cfg, cfgfile)
        curdir = os.path.dirname(cfgfile)
        print(cfg.sim_config.mjcf_path)

        cfg.sim_config.model_path = cfg.sim_config.model_path.format(
            workspace_dir=workspace_dir,
            experiment_name=experiment_name,
            currentdir=curdir
        )
        cfg.sim_config.param_path = cfg.sim_config.param_path.format(
            workspace_dir=workspace_dir,
            experiment_name=experiment_name,
            currentdir=curdir
        )
        # use model and param given by user
        if args.load_model is not None and args.model_param is not None:
            cfg.sim_config.model_path = args.load_model
            cfg.sim_config.param_path = args.model_param


    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()
    # rospy.init_node("publisher")
    # pub = rospy.Publisher("/pubtest/Acc", Float32, queue_size=10)

    policy = torch.jit.load(cfg.sim_config.model_path)
    param = torch.load(cfg.sim_config.param_path)

    node = HumaRobotSimRos('fdhumarobot_simros')

    node.run_mujoco(policy, param, cfg)
    # node.destroy_node()
    # rclpy.shutdown()
