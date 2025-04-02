# Copyright (c) 2024-2024, fudeRobot All rights reserved.
#
# References:
#   this module adapted from humanoid-gym [Open Source Project](https://github.com/roboterax/humanoid-gym.git)


import os
import shutil


from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *
from humanoid.utils import get_args, task_registry
import humanoid.tools.common as tool_util


def save_sim2sim_cfg(log_dir, mjcf_file, env):
    if env.num_actions > 12:
        from humanoid.config.sim2siml_config import Sim2simCfg
    else:
        from humanoid.config.sim2sim_config import Sim2simCfg
    cfg = Sim2simCfg()
    cfg.sim_config.mjcf_path = mjcf_file
    tool_util.save_class_to_yaml_file(cfg, os.path.join(log_dir, tool_util.TRAINCOPY_SIM2SIMCFG_FILENAME))


def save_asset_file(env_cfg, log_dir):
    src_asset_file = env_cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    asset_cp_dir = tool_util.get_asset_path_train_backup(src_asset_file)

    load_run = os.path.basename(log_dir)  # dist_asset_prefix = ${load_run}

    filename = os.path.basename(src_asset_file)  # filename:  xxx.urdf
    name, ext = os.path.splitext(filename)

    # save urdf asset file, which be used at play
    dist_asset_file = os.path.join(asset_cp_dir, f'{name}_{load_run}{ext}')
    os.makedirs(asset_cp_dir, exist_ok=True)
    shutil.copy(src_asset_file, dist_asset_file)  # meshes are located here
    env_cfg.asset.file = dist_asset_file

    # backup urdf file to mode_path
    os.makedirs(log_dir, exist_ok=True)
    model_path_asset_file = os.path.join(log_dir, tool_util.TRAINCOPY_ASSET_FILENAME)
    shutil.copy(src_asset_file, model_path_asset_file)

    # backup mjcf file to model path
    robot_assets_path = os.path.dirname(os.path.dirname(src_asset_file))
    src_mjcf_file = os.path.join(robot_assets_path, 'mjcf', f'{name}.xml')
    model_path_mjcf_file = os.path.join(log_dir, tool_util.TRAINCOPY_MJCF_FILENAME)
    shutil.copy(src_mjcf_file, model_path_mjcf_file)

    # save mjcf file, which be used at sim2sim
    dist_mjcf_file = tool_util.get_mjcf_filename_train_backup(src_asset_file, load_run)
    shutil.copy(src_mjcf_file, dist_mjcf_file)
    env_cfg.asset.mjcf_file = dist_mjcf_file

    return env_cfg.asset.file


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    save_asset_file(env_cfg, ppo_runner.log_dir)
    env_cfg_export_filepath = os.path.join(ppo_runner.log_dir, tool_util.ENV_CFG_FILENAME)
    train_cfg_export_filepath = os.path.join(ppo_runner.log_dir, tool_util.TRAIN_CFG_FILENAME)
    tool_util.save_class_to_yaml_file(env_cfg, env_cfg_export_filepath)
    tool_util.save_class_to_yaml_file(train_cfg, train_cfg_export_filepath)
    save_sim2sim_cfg(ppo_runner.log_dir, env_cfg.asset.mjcf_file, env)

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)


if __name__ == '__main__':
    try:
        args = get_args()
    except Exception as e:
        print(e)
        import sys
        sys.exit(0)

    train(args)
