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


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class FdBotCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 61 # 64
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 85 # 88
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False

        queue_len_obs = 3
        queue_len_act = 3
        obs_latency = [5, 20]
        act_latency = [5, 20]

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/T1Bot/urdf/t1_s.urdf'

        name = "FdBot"
        foot_name = "ankleP"
        knee_name = "knee"
        # torso_imu_name = "torsoP_Link"
        # hand_name = "elbowR"
        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        collapse_fixed_joints = True
        test_mode = False
        pos_soft_limit_range_scale = 0.9

        class test_gait:
            signal_type = 1
            offset = 0.2
            pp = 0.15
            cycle_time = 1.5
            joint_ids = [0, 6]

        if test_mode:
            fix_base_link = True
        else:
            fix_base_link = False

        hip_y_friction = 0.08
        hip_r_friction = 0.04
        hip_p_friction = 0.08
        knee_friction = 0.085
        ankle_r_friction = 1.0
        ankle_p_friction = 1.0

        # damping
        hip_y_damping = 2.5
        hip_r_damping = 15.
        hip_p_damping = 17.5
        knee_damping = 6.5
        ankle_r_damping = 3.
        ankle_p_damping = 12.

        # armature
        hip_y_armature = 1.3
        hip_r_armature = 0.3
        hip_p_armature = 2.5
        knee_armature = 1.6
        ankle_r_armature = 0.45
        ankle_p_armature = 0.9

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        mesh_type = 'trimesh'
        curriculum = True
        # rough terrain only:
        measure_heights = False
        static_friction = 1.0
        dynamic_friction = 1.0
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0, 0, 0.2, 0, 0, 0.5, 0.3]
        restitution = 0

    class noise:
        add_noise = True
        noise_level = 1.0    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            torque = 0.1
            force = 0.
            torso_ang_vel = 0.1
            torso_quat = 0.03
            lin_acc = 0.1
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.90]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'leg_left_hipy': 0.,    # 0
            'leg_left_hipr': 0.,    # 1
            'leg_left_hipp': 0.,    # 2
            'leg_left_knee': 0.,    # 3
            'leg_left_ankler': 0.,  # 4
            'leg_left_anklep': 0.,  # 5
            'leg_right_hipy': 0.,   # 6
            'leg_right_hipr': 0.,   # 7
            'leg_right_hipp': 0.,   # 8
            'leg_right_knee': 0.,   # 9
            'leg_right_ankler': 0.,   # 10
            'leg_right_anklep': 0.,   # 11
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'hipy': 200.0, 'hipr': 325.0, 'hipp': 1450.0,
                     'knee': 550.0, 'ankler': 130, 'anklep': 310.,
                    }
        damping = {'hipy': 7.5, 'hipr': 12., 'hipp': 75.,
                   'knee': 20., 'ankler': 3., 'anklep': 3.5,
                   }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # 50hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 200 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 1.0  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**24  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        
        randomize_restitution = True
        restitution_range = [0.0, 0.4]
        
        randomize_dof_friction = True
        dof_friction_range = [-0.5, 0.5]
        
        randomize_dof_damping = True
        dof_damping_range = [-0.5, 0.5]
        
        randomize_dof_armature = True
        dof_armature_range = [-0.5, 0.5]
        
        randomize_cycle_time = True
        use_command_cycle_time = False
        cycle_time_range = [0.5, 0.8]
        
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  
        damping_multiplier_range = [0.8, 1.2]   
        
        randomize_calculated_torque = True
        torque_multiplier_range = [0.8, 1.2]
        
        randomize_link_mass = True
        multiplied_link_mass_range = [0.8, 1.2]
        
        
        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035] # Offset to add to the motor angles
        
        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [1, 10]
        
        add_obs_latency = True # no latency for obs_action
        randomize_obs_motor_latency = True
        randomize_obs_imu_latency = True
        range_obs_motor_latency = [1, 10]
        range_obs_imu_latency = [1, 10]
        
        
        push_robots = True
        push_interval_s = 12  # 4
        max_push_vel_xy = 0.6  # 0.4 # 2.0
        max_push_ang_vel = 0.3
        push_duration = 1  # 1
        push_curriculum_start_step = 2000 * 60
        push_curriculum_common_step = 10000 * 60
        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02

        randomize_body_com = True
        added_body_com_range = [-0.06, 0.06]  # 1cm -> 0.02
        added_leg_com_range = [-0.05, 0.05]  # 0.5cm -> 0.005

        randomize_body_inertia = True
        scaled_body_inertia_range = [0.90, 1.1]  # %5 error

        default_joint_angles_range = {  # = target angles [rad] when action = 0.0
            # 'leg_left_hipy': [-0.1, 0.1],    # 0
            # 'leg_left_hipr': [-0.2, 0.8],    # 1
            # 'leg_left_hipp': [-0.5, 1.0],    # 2
            # 'leg_left_knee': [0.0, 0.5],    # 3
            # 'leg_left_ankler': [-0.1, 0.1],  # 4
            # 'leg_left_anklep': [-0.01, 0.5],  # 5

            # 'leg_right_hipy': [-0.1, 0.1],   # 6
            # 'leg_right_hipr': [-0.2, 0.8],   # 7
            # 'leg_right_hipp': [-0.5, 1.0],   # 8
            # 'leg_right_knee': [-0.0, 0.5],   # 9
            # 'leg_right_ankler': [-0.1, 0.1],  # 10
            # 'leg_right_anklep': [-0.01, 0.5],  # 11
            
            'leg_left_hipy': [-0.1, 0.1],    # 0
            'leg_left_hipr':  [-0.1, 0.1],    # 1
            'leg_left_hipp':  [-0.1, 0.1],    # 2
            'leg_left_knee':  [-0.1, 0.1],    # 3
            'leg_left_ankler':  [-0.1, 0.1],  # 4
            'leg_left_anklep':  [-0.1, 0.1],  # 5

            'leg_right_hipy':  [-0.1, 0.1],   # 6
            'leg_right_hipr':  [-0.1, 0.1],   # 7
            'leg_right_hipp':  [-0.1, 0.1],   # 8
            'leg_right_knee':  [-0.1, 0.1],   # 9
            'leg_right_ankler':  [-0.1, 0.1],  # 10
            'leg_right_anklep':  [-0.1, 0.1],  # 11
        }

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        max_curriculum = 1.
        curriculum = True
        class ranges:
            lin_vel_x = [-0.5, 1.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1., 1.]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        # base_height_target = 0.87
        min_dist = 0.16
        max_dist = 1.0
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale_knee = 0.8    # rad 0.25 0.2
        target_joint_pos_scale = 0.3    # rad 0.25 0.2
        target_feet_height = 0.15
        cycle_time = 0.64                # sec [0.32-0.64]
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 400  # forces above this value are penalized

        class scales:
            # reference motion tracking
            joint_pos = 1.8
            feet_contact_number = 1.2
            # gait
            foot_slip = -0.1
            feet_distance = 0.2
            # contact
            feet_contact_forces = -0.02
            # vel tracking
            tracking_lin_vel = 2.
            tracking_ang_vel = 1.2
            low_speed = 0.2
            track_vel_hard = 0.5
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            # base pos
            orientation = 1.0
            base_acc = 0.2
            # energy
            action_smoothness = -0.003
            torques = -1e-6
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.
            # stand
            zero_stand = 2.0
            # foot
            feet_orientation = 0.5
            feet_clearance = 1.0
            ankle_torques = 0.5
            # leg
            default_joint_pos = 0.2
            dof_pos_limits = -10.0
            actions_limits = -1.0

    class normalization:
        class obs_scales:
            lin_vel = 2.  # cmd_x/y_linear_scale torso_linear_vel_scale
            ang_vel = 1.   # imu_angle_vel_scale torso_ang_vel_scale
            waist_ang_vel = 1.0   # imu_angle_vel_scale torso_ang_vel_scale
            dof_pos = 1.   # joint position scale
            dof_vel = 0.05  # joint velocities scale
            quat = 1.0      # imu orientation scale  default 1
            torque = 0.     # ankle joint torque scale
            force = 0.  # 1.  # feet contact force scale
            torso_ang_vel = 0.  # torso angular velocity scale
            torso_quat = 0.  # torso orientation scale
            lin_acc = 0.  # imu linear acc scale
            height_measurements = 5.0  #
        clip_observations = 18.
        clip_actions = 18.

class FdBotCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 20001  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'FdBot_ppo'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
