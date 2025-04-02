# Copyright (c) 2024-2024, fudeRobot All rights reserved.
#

class Sim2simCfg():
    """Sim configuration."""
    class sim_config:
        # path: supported placeholder : workspace_dir  experiment_name(such as FdBot_ppo)  currentdir(path of the sim2sim_cfg file)
        mjcf_path = '{workspace_dir}/resources/robots/T1Bot/mjcf/t1.xml'
        model_path = '{workspace_dir}/logs/{experiment_name}/exported/policies/policy.pt'
        param_path = '{workspace_dir}/logs/{experiment_name}/exported/policies/params.pt'

        sim_duration = 10000.0  # 100000.0
        dt = 0.001      # 实时
        decimation = 10  # 分频

    class robot_config:
        kps = [200., 325., 1450., 550, 130., 310.,
               200., 325., 1450., 550., 130., 310.,
               ]
        kds = [7.5, 12., 75., 20., 3., 3.5,
               7.5, 12., 75., 20., 3., 3.5,
               
               ]

        tau_limit = [90, 150, 200, 250, 40, 80,
                     90, 150, 200, 250, 40, 80,
                     ]
