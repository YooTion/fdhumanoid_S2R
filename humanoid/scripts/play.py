# Copyright (c) 2024-2024, fudeRobot All rights reserved.
#
# References:
#   this module adapted from humanoid-gym [Open Source Project](https://github.com/roboterax/humanoid-gym.git)

import os
import cv2
import numpy as np
import shutil
import yaml
from pynput import keyboard
import threading

from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.utils import  get_args, export_policy_as_jit, task_registry, Logger, get_load_path
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm

import humanoid.tools.common as tool_util

# 全局变量存储速度命令
current_lin_vel = 0.0
current_lat_vel = 0.0
lock = threading.Lock()  # 线程锁确保数据安全

def on_press(key):
    global current_lin_vel, current_lat_vel
    try:
        with lock:
            if key.char == 'w':
                current_lin_vel = 1.0
            elif key.char == 's':
                current_lin_vel = -1.0
            elif key.char == 'a':
                current_lat_vel = 1.0
            elif key.char == 'd':
                current_lat_vel = -1.0
    except AttributeError:
        pass

def on_release(key):
    global current_lin_vel, current_lat_vel
    try:
        with lock:
            if key.char in ('w', 's'):
                current_lin_vel = 0.0
            elif key.char in ('a', 'd'):
                current_lat_vel = 0.0
    except AttributeError:
        pass


def get_model_path(experiment_name, load_run, checkpoint, log_root="default"):
    if log_root == 'default':
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', experiment_name)
    resume_path = get_load_path(log_root, load_run=load_run, checkpoint=checkpoint)
    return os.path.dirname(resume_path)


def policy_exported_param(env):
    dof_names = []
    obs_index_qd = 5 + env.cfg.env.num_actions
    obs_index_action = 5 + 2 * env.cfg.env.num_actions
    obs_index_imu_waist_angvel = 5 + 3 * env.cfg.env.num_actions
    obs_index_imu_waist_orientation = 5 + 3 * env.cfg.env.num_actions + 3
    obs_index_torque = 5 + 3 * env.cfg.env.num_actions + 6
    obs_index_forcesensor = 5 + 4 * env.cfg.env.num_actions + 6
    obs_index_imu_waist_linearacc = -1
    obs_index_imu_torso_angvel = -1
    obs_index_imu_torso_orientation = -1
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

        if name == "torsoy":
            dof_names.append("joint_torsoy")
        if name == "torsor":
            dof_names.append("joint_torsor")
        if name == "torsop":
            dof_names.append("joint_torsop")
        if name == "arm_left_humeraly":
            dof_names.append("joint_arm_left_humeraly")
        if name == "arm_left_humeralr":
            dof_names.append("joint_L_humeralR")
        if name == "arm_left_wristr":
            dof_names.append("joint_arm_left_elbow")

        if name == "arm_right_humeraly":
            dof_names.append("joint_arm_right_humeraly")
        if name == "arm_right_humeralr":
            dof_names.append("joint_R_humeralR")
        if name == "arm_right_wristr":
            dof_names.append("joint_arm_right_elbow")

    policy_param = {
        "ref_cycle_type": 0,
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
        "obs_scales_imu_waist_angvel": env.cfg.normalization.obs_scales.waist_ang_vel,
        "obs_scales_imu_waist_orientation": env.cfg.normalization.obs_scales.quat,
        "obs_scales_torque": env.cfg.normalization.obs_scales.torque,
        "obs_scales_imu_waist_linearacc": env.cfg.normalization.obs_scales.lin_acc,
        "obs_scales_force": env.cfg.normalization.obs_scales.force,
        "obs_scales_imu_torso_angvel": env.cfg.normalization.obs_scales.torso_ang_vel,
        "obs_scales_imu_torso_orientation": env.cfg.normalization.obs_scales.torso_quat,

        "obs_index_q": 5,
        "obs_index_qd": obs_index_qd,
        "obs_index_action": obs_index_action,
        "obs_index_imu_waist_angvel": obs_index_imu_waist_angvel,
        "obs_index_imu_waist_orientation": obs_index_imu_waist_orientation,
        "obs_index_torque": obs_index_torque,
        "obs_index_forcesensor": obs_index_forcesensor,
        "obs_index_imu_waist_linearacc": obs_index_imu_waist_linearacc,
        "obs_index_imu_torso_angvel": obs_index_imu_torso_angvel,
        "obs_index_imu_torso_orientation": obs_index_imu_torso_orientation,
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
    return policy_param


def export_policy_param_file(env, filepath):
    params_dict = {
        "rl": policy_exported_param(env)
    }
    param_yaml = yaml.dump(params_dict)
    yaml_path = os.path.join(filepath, 'params.yaml')
    with open(yaml_path, 'w') as f:
        f.write(param_yaml)
    return yaml_path


def export_policy_param_pt(env, filepath):
    pt_path = os.path.join(filepath, 'params.pt')
    torch.save(policy_exported_param(env), pt_path)


def export_sim2simconfig(model_path, export_path, mjcf_path, env_cfg):
    cfg_path = os.path.join(model_path, tool_util.TRAINCOPY_SIM2SIMCFG_FILENAME)
    dist_file = os.path.join(export_path, tool_util.SIM2SIMCFG_FILENAME)
    if env_cfg.env.num_actions > 12:
        import humanoid.config.sim2siml_config as sim2simcfg
    else:
        import humanoid.config.sim2sim_config as sim2simcfg
    cfg = sim2simcfg.Sim2simCfg()
    tool_util.update_class_from_yaml_file(cfg, cfg_path)
    cfg.sim_config.mjcf_path = mjcf_path
    cfg.sim_config.model_path = '{currentdir}/policy.pt'
    cfg.sim_config.param_path = '{currentdir}/params.pt'
    tool_util.save_class_to_yaml_file(cfg, dist_file)
    return dist_file


def export_assets(src_asset_file, model_mjcf_file, load_run):
    dist_mjcf_file = tool_util.get_mjcf_filename_train_backup(src_asset_file, load_run)
    if os.path.exists(model_mjcf_file):
        shutil.copy2(model_mjcf_file, dist_mjcf_file)
        print(f'copy: {model_mjcf_file}->{dist_mjcf_file}')
    return dist_mjcf_file


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    src_asset_file = env_cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

    if hasattr(args, 'experiment_name') and args.experiment_name is not None:
        experiment_name = args.experiment_name
    else:
        experiment_name = train_cfg.runner.experiment_name

    if args.load_run is not None:
        load_run = args.load_run
    else:
        load_run = train_cfg.runner.load_run

    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    else:
        checkpoint = train_cfg.runner.checkpoint

    #  load and override configs which exported at training if exists
    modepath = get_model_path(experiment_name, load_run, checkpoint)
    saved_env_cfg_file = os.path.join(modepath, tool_util.ENV_CFG_FILENAME)
    saved_train_cfg_file = os.path.join(modepath, tool_util.TRAIN_CFG_FILENAME)
    if os.path.exists(saved_env_cfg_file):
        tool_util.update_class_from_yaml_file(env_cfg, saved_env_cfg_file)
        if not os.path.exists(env_cfg.asset.file):
            print(f'can not found asset file:{env_cfg.asset.file}, (env_cfg.asset.file) in: {saved_env_cfg_file}')
            return
    if os.path.exists(saved_train_cfg_file):
        tool_util.update_class_from_yaml_file(train_cfg, saved_train_cfg_file)

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)

    env_cfg.sim.max_gpu_contact_pairs = 2**10
    env_cfg.terrain.mesh_type = 'trimesh'
    # env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = True
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
        policy_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        policy_file = export_policy_as_jit(ppo_runner.alg.actor_critic, policy_path)
        print('Exported policy as jit script to: ', policy_file)
        param_file = export_policy_param_file(env, policy_path)
        print('Exported policy param to: ', param_file)
        export_policy_param_pt(env, policy_path)

        if hasattr(env_cfg.asset, 'mjcf_file') and os.path.exists(env_cfg.asset.mjcf_file):
            mjcf_file = env_cfg.asset.mjcf_file
            print('Use mjcf file: ', mjcf_file)
        else:
            model_path_mjcf_file = os.path.join(modepath, tool_util.TRAINCOPY_MJCF_FILENAME)
            mjcf_file = export_assets(src_asset_file, model_path_mjcf_file, load_run)
            print('Exported mjcf file to: ', mjcf_file)

        simcfg_file = export_sim2simconfig(modepath, policy_path, mjcf_file, env_cfg)
        print('Exported sim2sim config file to: ', simcfg_file)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 2  # which joint is used for logging
    stop_state_log = 10000  # number of steps before plotting states
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

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        # dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        # video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))

    for i in range(stop_state_log):
        actions = policy(obs.detach())
        
        # 使用线程安全的数值
        with lock:
            env.commands[:, 0] = current_lin_vel  # 线速度
            env.commands[:, 1] = current_lat_vel  # 侧向速度
            env.commands[:, 2] = 0.0
            env.commands[:, 3] = 0.0
        
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # video.write(img[..., :3])
        logger.log_states(
            {
                'dof_pos': env.ref_dof_pos[0, 2].item(),
                'dof_pos_target': env.ref_dof_pos[0, 3].item(),
                'dof_vel': env.ref_dof_pos[0, 8].item(),
                'dof_vel_target': env.ref_dof_pos[0, 9].item(),
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
    # logger.plot_states()
    # logger._plot()
    # if RENDER:
    #     video.release()


if __name__ == '__main__':
    EXPORT_POLICY = True
    RENDER = False  # True
    FIX_COMMAND = True
    # 启动键盘监听线程
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    args = get_args()
    play(args)
