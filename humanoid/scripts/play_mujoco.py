# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
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

import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *
import torch
from tqdm import tqdm
import yaml

import sys
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(workspace_dir)  # append workspace directory to PYTHONPATH


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands."""
    return (target_q - q) * kp + (target_dq - dq) * kd


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
    torque_foot[0] = -data.sensor('LeftFootForceSensor_tsensor').data.astype(np.double)[0]
    torque_foot[1] = data.sensor('RightFootForceSensor_tsensor').data.astype(np.double)[0]
    torque_foot[2] = data.sensor('LeftFootForceSensor_tsensor').data.astype(np.double)[1]
    torque_foot[3] = data.sensor('RightFootForceSensor_tsensor').data.astype(np.double)[1]
    force_foot = np.zeros([2])
    force_foot[0] = -data.sensor('LeftFootForceSensor_fsensor').data.astype(np.double)[2]
    force_foot[1] = -data.sensor('RightFootForceSensor_fsensor').data.astype(np.double)[2]

    force_foot[force_foot < 0] = 0
    force_foot = force_foot > 10
    # print(force_foot)
    lin_vel = data.sensor('linear-velocity').data.astype(np.double)
    # print("force_foot:", force_foot)
    # global q_obs
    # global dq_obs
    # global quat_obs
    # global omega_obs
    # print("当前传感器值:", omega)
    # q_obs.append(q)
    # dq_obs.append(dq)
    # quat_obs.append(quat)
    # omega_obs.append(omega)
    # q_obs.popleft()
    # dq_obs.popleft()
    # quat_obs.popleft()
    # omega_obs.popleft()

    # q = q_obs[0]
    # dq = dq_obs[0]
    # quat = quat_obs[0]
    # omega = omega_obs[0]
    # print("延迟缓存大小:", len(omega_obs))
    # print("延迟后传感器值:", omega)
    # print("延迟缓存末端值:", omega_obs[-1])
    return (q, dq, quat, v, omega, gvec, lin_acc, torque_foot, lin_vel, force_foot)


def play(args, cfg):
    # mujoco
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
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

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.joint_angle_noise = 0.
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
        dof_names = []
        obs_index_qd = 5 + env.cfg.env.num_actions
        obs_index_action = 5 + 2 * env.cfg.env.num_actions
        obs_index_imu_waist_angvel = 5 + 3 * env.cfg.env.num_actions
        obs_index_imu_waist_orientation = 5 + 3 * env.cfg.env.num_actions + 3
        obs_index_torque = 5 + 3 * env.cfg.env.num_actions + 6
        obs_index_forcesensor = 5 + 4 * env.cfg.env.num_actions + 6
        obs_index_imu_waist_linearacc = -1
        obs_index_imu_torso_orientation = -1
        obs_index_imu_torso_angvel = -1
        obs_index_imu_torso_linearacc = -1

        for name in env.dof_names:
            if name == "leg_left_hipy":
                dof_names.append("joint_L_hipY")
            if name == "leg_left_hipr":
                dof_names.append("joint_L_hipR")
            if name == "leg_left_hipp":
                dof_names.append("joint_L_hipP")
            if name == "leg_left_knee":
                dof_names.append("joint_L_knee")
            if name == "leg_left_ankler":
                dof_names.append("joint_L_ankleR")
            if name == "leg_left_anklep":
                dof_names.append("joint_L_ankleP")
            if name == "leg_right_hipy":
                dof_names.append("joint_R_hipY")
            if name == "leg_right_hipr":
                dof_names.append("joint_R_hipR")
            if name == "leg_right_hipp":
                dof_names.append("joint_R_hipP")
            if name == "leg_right_knee":
                dof_names.append("joint_R_knee")
            if name == "leg_right_ankler":
                dof_names.append("joint_R_ankleR")
            if name == "leg_right_anklep":
                dof_names.append("joint_R_ankleP")

        params_dict = {
            "rl":
                {
                    "ref_cycle_type": env.cfg.rewards.scales.zero_stand > 0,
                    "dof_names": dof_names,
                    "num_actions": env.cfg.env.num_actions,
                    "frame_stack": env.cfg.env.frame_stack,
                    "num_single_obs": env.cfg.env.num_single_obs,
                    "dt": env.dt,
                    "cycle_time": env.cfg.rewards.cycle_time,
                    "clip_observations": env.cfg.normalization.clip_observations,
                    "num_observations": env.cfg.env.num_observations,
                    "clip_actions": env.cfg.normalization.clip_actions,
                    "action_scale": env.cfg.control.action_scale,
                    "num_forcesensor": torch.numel(env.filter_forces_z[0]),

                    "obs_scales_dof_pos": env.cfg.normalization.obs_scales.dof_pos,
                    "obs_scales_dof_vel": env.cfg.normalization.obs_scales.dof_vel,
                    "obs_scales_lin_vel": env.cfg.normalization.obs_scales.lin_vel,
                    "obs_scales_ang_vel": env.cfg.normalization.obs_scales.ang_vel,
                    "obs_scales_imu_waist_angvel": env.cfg.normalization.obs_scales.ang_vel,
                    "obs_scales_imu_waist_orientation": env.cfg.normalization.obs_scales.quat,
                    "obs_scales_torque": env.cfg.normalization.obs_scales.torque,
                    "obs_scales_imu_waist_linearacc": env.cfg.normalization.obs_scales.lin_acc,
                    "obs_scales_force": env.cfg.normalization.obs_scales.force,
                    "obs_scales_imu_torso_angvel": -1,
                    "obs_scales_imu_torso_orientation": -1,

                    "obs_index_q": 5,
                    "obs_index_qd": obs_index_qd,
                    "obs_index_action": obs_index_action,
                    "obs_index_imu_waist_angvel": obs_index_imu_waist_angvel,
                    "obs_index_imu_waist_orientation": obs_index_imu_waist_orientation,
                    "obs_index_torque": obs_index_torque,
                    "obs_index_forcesensor": obs_index_forcesensor,
                    "obs_index_imu_waist_linearacc": obs_index_imu_waist_linearacc,
                    "obs_index_imu_torso_orientation": obs_index_imu_torso_orientation,
                    "obs_index_imu_torso_angvel": obs_index_imu_torso_angvel,
                    "obs_index_imu_torso_linearacc": obs_index_imu_torso_linearacc,


                    "has_obs_torque": (env.cfg.env.num_single_obs > obs_index_torque and obs_index_torque != -1),
                    "has_obs_imu_waist_linearacc":
                        (env.cfg.env.num_single_obs > obs_index_imu_waist_linearacc and obs_index_imu_waist_linearacc != -1),
                    "has_obs_forcesensor":
                        (env.cfg.env.num_single_obs > obs_index_forcesensor and obs_index_forcesensor != -1),
                    "has_obs_imu_torso":
                        (env.cfg.env.num_single_obs > obs_index_imu_torso_orientation and obs_index_imu_torso_orientation != -1),
                    "has_obs_imu_torso_linearacc":
                        (env.cfg.env.num_single_obs > obs_index_imu_torso_linearacc and obs_index_imu_torso_linearacc != -1)
                }
        }
        param_yaml = yaml.dump(params_dict)
        yaml_path = os.path.join(path, "params.yaml")
        with open(yaml_path, 'w') as f:
            f.write(param_yaml)

        pt_path = os.path.join(path, "params.pt")
        torch.save(
            {
                "ref_cycle_type": env.cfg.rewards.scales.zero_stand > 0,
                "dof_names": dof_names,
                "num_actions": env.cfg.env.num_actions,
                "frame_stack": env.cfg.env.frame_stack,
                "num_single_obs": env.cfg.env.num_single_obs,
                "dt": env.dt,
                "cycle_time": env.cfg.rewards.cycle_time,
                "clip_observations": env.cfg.normalization.clip_observations,
                "num_observations": env.cfg.env.num_observations,
                "clip_actions": env.cfg.normalization.clip_actions,
                "action_scale": env.cfg.control.action_scale,
                "num_forcesensor": torch.numel(env.filter_forces_z[0]),

                "obs_scales_dof_pos": env.cfg.normalization.obs_scales.dof_pos,
                "obs_scales_dof_vel": env.cfg.normalization.obs_scales.dof_vel,
                "obs_scales_lin_vel": env.cfg.normalization.obs_scales.lin_vel,
                "obs_scales_ang_vel": env.cfg.normalization.obs_scales.ang_vel,
                "obs_scales_imu_waist_angvel": env.cfg.normalization.obs_scales.ang_vel,
                "obs_scales_imu_waist_orientation": env.cfg.normalization.obs_scales.quat,
                "obs_scales_torque": env.cfg.normalization.obs_scales.torque,
                "obs_scales_imu_waist_linearacc": env.cfg.normalization.obs_scales.lin_acc,
                "obs_scales_force": env.cfg.normalization.obs_scales.force,
                "obs_scales_imu_torso_angvel": -1,
                "obs_scales_imu_torso_orientation": -1,

                "obs_index_q": 5,
                "obs_index_qd": obs_index_qd,
                "obs_index_action": obs_index_action,
                "obs_index_imu_waist_angvel": obs_index_imu_waist_angvel,
                "obs_index_imu_waist_orientation": obs_index_imu_waist_orientation,
                "obs_index_torque": obs_index_torque,
                "obs_index_forcesensor": obs_index_forcesensor,
                "obs_index_imu_waist_linearacc": obs_index_imu_waist_linearacc,
                "obs_index_imu_torso_orientation": obs_index_imu_torso_orientation,
                "obs_index_imu_torso_angvel": obs_index_imu_torso_angvel,
                "obs_index_imu_torso_linearacc": obs_index_imu_torso_linearacc,


                "has_obs_torque": (env.cfg.env.num_single_obs > obs_index_torque and obs_index_torque != -1),
                "has_obs_imu_waist_linearacc":
                    (env.cfg.env.num_single_obs > obs_index_imu_waist_linearacc and obs_index_imu_waist_linearacc != -1),
                "has_obs_forcesensor":
                    (env.cfg.env.num_single_obs > obs_index_forcesensor and obs_index_forcesensor != -1),
                "has_obs_imu_torso":
                    (env.cfg.env.num_single_obs > obs_index_imu_torso_orientation and obs_index_imu_torso_orientation != -1),
                "has_obs_imu_torso_linearacc":
                    (env.cfg.env.num_single_obs > obs_index_imu_torso_linearacc and obs_index_imu_torso_linearacc != -1)
            },
            pt_path,
        )
    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 2  # which joint is used for logging
    stop_state_log = 500  # number of steps before plotting states
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        # # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        # experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        # # dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        # if not os.path.exists(video_dir):
        #     os.mkdir(video_dir)
        # if not os.path.exists(experiment_dir):
        #     os.mkdir(experiment_dir)
        # video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))
    target_q = np.zeros(12)
    for i in tqdm(range(stop_state_log)):

        actions = policy(obs.detach())  # * 0.
        target_q[env.cfg.asset.test_gait.joint_ids[0]] = env.ref_pos[0].item()
        q, dq, quat, v, omega, gvec, lin_acc, torque_foot, lin_vel, force_foot = get_obs(data)
        for _ in range(10):
            q, dq, quat, v, omega, gvec, lin_acc, torque_foot, lin_vel, force_foot = get_obs(data)
            q = q[-env.cfg.env.num_actions:]
            dq = dq[-env.cfg.env.num_actions:]
            target_dq = np.zeros_like(q)
            tau = pd_control(target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)  # Calc torques
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
            data.ctrl = tau
            mujoco.mj_step(model, data)
        viewer.sync()

        if FIX_COMMAND:
            env.commands[:, 0] = 0.5  # 1.0
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.
            env.commands[:, 3] = 0.

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # video.write(img[..., :3])
        if env.cfg.asset.test_mode:
            logger.log_states(
                {
                    'dof_pos': env.dof_pos[0, env.cfg.asset.test_gait.joint_ids[0]].item(),
                    'dof_ref_pos': env.ref_pos[0].item(),
                    'mujoco_pos': q[env.cfg.asset.test_gait.joint_ids[0]],
                    'dof_vel': env.dof_vel[0, env.cfg.asset.test_gait.joint_ids[0]].item(),
                    'mujoco_vel': dq[env.cfg.asset.test_gait.joint_ids[0]]
                }
            )
        else:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.foot_tue_xy[0, 0].item(),
                    'dof_vel': env.foot_tue_xy[0, 2].item(),
                    'dof_torque': env.filter_forces_z[0, 0].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        # ====================== Log states ======================
        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes > 0:
                logger.log_rewards(infos["episode"], num_episodes)

    logger.print_rewards()

    if env.cfg.asset.test_mode:
        logger._plot_test()
    else:
        logger.plot_states()
        logger._plot()
    viewer.close()
    # if RENDER:
    #     video.release()


if __name__ == '__main__':
    EXPORT_POLICY = False
    RENDER = True
    FIX_COMMAND = True
    args = get_args()

    class Sim2simCfg():
        """Sim configuration."""
        class sim_config:
            # if args.terrain:
            #     mujoco_model_path = '../resources/robots/FdBot/mjcf/fdhumanoid_s-terrain.xml'
            # else:
            mujoco_model_path = f'{workspace_dir}/resources/robots/T1Bot/mjcf/t1_test.xml'
            sim_duration = 100000.0  # 100000.0
            dt = 0.001      # 实时
            decimation = 10  # 分频

        class robot_config:
            kps = np.array([50., 325., 1450., 550, 80., 200.,
                            50., 325., 1450., 550., 80., 200.
                            # , 350, 200, 200, 200, 200
                            ], dtype=np.double)
            kds = np.array([0., 12., 75., 20., 3.5, 4.5,
                            0., 12., 75., 20., 3.5, 4.5
                            # , 10, 10, 10, 10, 10
                            ], dtype=np.double)
            # kds = np.zeros([12])
            tau_limit = np.array([90, 150, 200, 250, 40, 80,
                                  90, 150, 200, 250, 40, 80], dtype=np.double)  # 200. * np.ones(12,

    play(args, Sim2simCfg())
