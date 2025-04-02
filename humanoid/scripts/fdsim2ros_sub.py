# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import numpy as np
import mujoco, mujoco_viewer
import mujoco.viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import time


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import  Imu, JointState
from control_msgs.msg import JointTrajectoryControllerState
import threading

from humanoid.tools.timer import Timer
import os
import sys
workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(workspace_dir)  # append workspace directory to PYTHONPATH


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


def get_obs(data):
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
        src_force_foot = force_foot.copy()

        force_foot[force_foot < 0] = 0
        force_foot = force_foot > 10
        # print(force_foot)
        lin_vel = data.sensor('linear-velocity').data.astype(np.double)

        return (q, dq, quat, v, omega, gvec, lin_acc, torque_foot, lin_vel, force_foot, src_force_foot)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_mujoco(cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    n_joints = model.njnt
    for i in range(n_joints):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"关节 {i}: {joint_name}")

    viewer = mujoco.viewer.launch_passive(model, data)
    target_q = np.zeros((cfg.sim_config.num_actions), dtype=np.double)
    count_lowlevel = 0
    ref_time = time.time()

    with Timer(f'[INFO]: Time taken for {cfg.sim_config.sim_duration:.2f}s simulation'):
        for _ in range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)):
            # Obtain an observation
            q, dq, quat, v, omega, gvec, lin_acc, torque_foot, lin_vel, force_foot, src_frc_foot = get_obs(data)
            q = q[-cfg.sim_config.num_actions:]
            dq = dq[-cfg.sim_config.num_actions:]
            eu_ang = quaternion_to_euler_array(quat)

            target_dq = np.zeros((cfg.sim_config.num_actions), dtype=np.double)
            # global _target_q
            mutex.acquire()
            if len(_target_q.joint_names) == cfg.sim_config.num_actions:
                target_q = get_target()
            mutex.release()
            # Generate PD control
            tau = pd_control(target_q, q, cfg.robot_config.kps,
                             target_dq, dq, cfg.robot_config.kds)  # Calc torques
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
            data.ctrl = tau

            mujoco.mj_step(model, data)
            ref_time += model.opt.timestep

            if count_lowlevel % 20 == 0:
                viewer.sync()

            if count_lowlevel % cfg.sim_config.decimation == 0:  # 100HZ time sync
                now_time = time.time()
                sec = int(now_time)
                nanoseconds = int((now_time - sec) * 1e9)

                #  接受实机命令给出反馈
                global _tranjstate_publisher
                msg1 = JointTrajectoryControllerState()
                msg1.header.stamp.sec = sec
                msg1.header.stamp.nanosec = nanoseconds
                msg1.joint_names = ["joint_L_hipY", "joint_L_hipR", "joint_L_hipP", "joint_L_knee", "joint_L_ankleR", "joint_L_ankleP",
                                    "joint_R_hipY", "joint_R_hipR", "joint_R_hipP", "joint_R_knee", "joint_R_ankleR", "joint_R_ankleP"]
                msg1.reference.positions = target_q
                msg1.reference.velocities = np.zeros(12)
                msg1.reference.effort = tau

                msg1.feedback.positions = q
                msg1.feedback.velocities = dq
                effort = np.zeros(12)
                effort[4] = torque_foot[0]
                effort[5] = torque_foot[2]
                effort[10] = torque_foot[1]
                effort[11] = torque_foot[3]
                msg1.feedback.effort = torque_foot

                msg1.error.effort = np.zeros(12)
                msg1.error.effort[5] = src_frc_foot[0]
                msg1.error.effort[11] = src_frc_foot[1]

                _tranjstate_publisher.publish(msg1)

                #  发布仿真状态供实机推理
                global _state_publisher
                msg2 = JointState()
                msg2.header.stamp.sec = sec
                msg2.header.stamp.nanosec = nanoseconds
                msg2.name = ["joint_L_hipY", "joint_L_hipR", "joint_L_hipP", "joint_L_knee", "joint_L_ankleR", "joint_L_ankleP",
                             "joint_R_hipY", "joint_R_hipR", "joint_R_hipP", "joint_R_knee", "joint_R_ankleR", "joint_R_ankleP"]
                msg2.position = q
                msg2.velocity = dq
                _state_publisher.publish(msg2)

                global _imu_publisher
                imu_data = Imu()
                imu_data.header.stamp.sec = sec
                imu_data.header.stamp.nanosec = nanoseconds
                imu_data.orientation.x = eu_ang[0]
                imu_data.orientation.y = eu_ang[1]
                imu_data.orientation.z = eu_ang[2]
                imu_data.angular_velocity.x = omega[0]
                imu_data.angular_velocity.y = omega[1]
                imu_data.angular_velocity.z = omega[2]
                imu_data.linear_acceleration.x = 0.
                imu_data.linear_acceleration.y = 0.
                imu_data.linear_acceleration.z = 0.
                _imu_publisher.publish(imu_data)

                cur_time = time.time()
                delta_time_s = ref_time - cur_time
                if delta_time_s > 0:
                    time.sleep(delta_time_s)
            count_lowlevel += 1
    viewer.close()


def get_target():
    global _target_q
    pos = []
    joint_names = ["joint_L_hipY", "joint_L_hipR", "joint_L_hipP", "joint_L_knee", "joint_L_ankleR", "joint_L_ankleP",
                   "joint_R_hipY", "joint_R_hipR", "joint_R_hipP", "joint_R_knee", "joint_R_ankleR", "joint_R_ankleP"]
    for j in joint_names:
        try:
            index = _target_q.joint_names.index(j)
            pos.append(_target_q.reference.positions[index])      
        except ValueError:
            return None
    return np.array(pos)


def target_callback(msg):
    global _target_q
    mutex.acquire()
    _target_q = msg
    mutex.release()


def spin_node(node):
    rclpy.spin(node)


if __name__ == '__main__':
    rclpy.init(args=None)
    node = Node("fdsim2sim")

    _tranjstate_publisher = node.create_publisher(JointTrajectoryControllerState,
                                                  "sim_ros/joint_states", 10)
    _imu_publisher = node.create_publisher(Imu, "sim_ros/imu", 10)

    #  测试实机推理时话题名修改为: joint_trajectory_controller/leggedrobot_rlsim_jointstate
    _state_publisher = node.create_publisher(JointState,
                                             "sim_ros/joint_trajectory_controller/leggedrobot_rlsim_jointstate", 10)

    _target_sub = node.create_subscription(JointTrajectoryControllerState,
                                           '/joint_trajectory_controller/controller_state', target_callback, 10)
    _target_q = JointTrajectoryControllerState()

    mutex = threading.Lock()

    spin_thread = threading.Thread(target=spin_node, args=(node,))
    spin_thread.start()

    class Sim2simCfg():
        class sim_config:
            mujoco_model_path = f'{workspace_dir}/resources/robots/T1Bot/mjcf/t1_test.xml'
            sim_duration = 100000.0
            dt = 0.001
            decimation = 10
            num_actions = 12

        class robot_config:
            kps = np.array([200., 325., 1450., 550, 130., 310.,
                            200., 325., 1450., 550., 130., 310.], dtype=np.double)
            kds = np.array([7, 12., 75., 23., 3., 3.5,
                            7., 12., 75., 23., 3., 3.5,], dtype=np.double)
            # kds = np.zeros([12])
            tau_limit = np.array([90, 150, 200, 250, 40, 80,
                                  90, 150, 200, 250, 40, 80,], dtype=np.double)
    run_mujoco(Sim2simCfg())
    node.destroy_node()
    rclpy.shutdown()
    spin_thread.join()
