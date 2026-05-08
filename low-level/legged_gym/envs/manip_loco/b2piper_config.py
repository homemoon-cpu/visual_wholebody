# B2+Piper configuration for ManipLoco training (Isaac Gym)
# Adapted from b1z1_config.py for B2 quadruped + Piper 6DOF arm + 2 gripper

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np


class B2PiperRoughCfg(LeggedRobotCfg):
    class goal_ee:
        num_commands = 3
        traj_time = [1, 3]
        hold_time = [0.5, 2]
        collision_upper_limits = [0.1, 0.15, -0.05]
        collision_lower_limits = [-0.6, -0.15, -0.55]
        underground_limit = -0.7
        num_collision_check_samples = 10
        command_mode = 'sphere'
        arm_induced_pitch = 0.38

        class sphere_center:
            x_offset = 0.2  # Relative to base (arm_base at x=0.2)
            y_offset = 0  # Relative to base
            z_invariant_offset = 0.68  # Relative to terrain (B2 standing 0.58 + arm_base z=0.1)

        class ranges:
            # Piper arm reach ~0.63m (link2=0.285 + link3=0.251 + link4-6+gripper≈0.091)
            # sphere_center距arm_base约0~0.1m, 所以pos_l上限≈0.55保守安全
            init_pos_start = [0.35, np.pi / 8, 0]
            init_pos_end = [0.5, 0, 0]
            pos_l = [0.25, 0.55]
            pos_p = [-1 * np.pi / 2.5, 1 * np.pi / 3]
            pos_y = [-1.0, 1.0]

            delta_orn_r = [-0.5, 0.5]
            delta_orn_p = [-0.5, 0.5]
            delta_orn_y = [-0.5, 0.5]
            final_tracking_ee_reward = 0.55

        sphere_error_scale = [1, 1, 1]
        orn_error_scale = [1, 1, 1]

    class noise:
        add_noise = False
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class commands:
        curriculum = True
        num_commands = 3
        resampling_time = 3.

        lin_vel_x_schedule = [0, 0.5]
        ang_vel_yaw_schedule = [0, 1]
        tracking_ang_vel_yaw_schedule = [0, 1]

        ang_vel_yaw_clip = 0.5
        lin_vel_x_clip = 0.2

        class ranges:
            lin_vel_x = [-0.8, 0.8]
            ang_vel_yaw = [-1.0, 1.0]

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class env:
        num_envs = 6144
        num_actions = 12 + 6  # 12 leg + 6 arm revolute (gripper not actuated by policy)
        num_torques = 12 + 6
        action_delay = 3
        num_gripper_joints = 2  # Piper has 2 prismatic gripper joints (arm_joint7, arm_joint8)
        num_proprio = 2 + 3 + 20 + 20 + 12 + 4 + 3 + 3 + 3  # 70
        # body_orn(2) + ang_vel(3) + joint_pos(20, no gripper) + joint_vel(20) + last_leg_action(12)
        # + foot_contacts(4) + commands(3) + ee_goal_cart(3) + ee_orn(3)
        # NOTE: num_proprio=70 vs b1z1's 66 because B2+Piper has 20 DOF (12 leg + 6 arm + 2 gripper)
        # but we only observe 18 joint positions (no gripper) => wait, let me re-check:
        # Actually following b1z1 pattern: 18 joints (12 leg + 6 arm, no gripper) for pos/vel
        # So: 2+3+18+18+12+4+3+3+3 = 66 (same as b1z1!)
        num_proprio = 2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3  # 66 (same as b1z1)
        num_priv = 5 + 1 + 12  # friction(5) + mass(1) + motor_strength(12 legs) = 18
        history_len = 10
        num_observations = num_proprio * (history_len + 1) + num_priv
        num_privileged_obs = None
        send_timeouts = True
        episode_length_s = 10
        reorder_dofs = True
        teleop_mode = False
        record_video = False
        stand_by = False
        observe_gait_commands = False
        frequencies = 2

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.58]  # B2 standing height ~0.58m
        default_joint_angles = {
            # B2 legs (same naming as URDF)
            'FL_hip_joint': 0.1,
            'FL_thigh_joint': 0.8,
            'FL_calf_joint': -1.5,

            'RL_hip_joint': 0.1,
            'RL_thigh_joint': 1.0,
            'RL_calf_joint': -1.5,

            'FR_hip_joint': -0.1,
            'FR_thigh_joint': 0.8,
            'FR_calf_joint': -1.5,

            'RR_hip_joint': -0.1,
            'RR_thigh_joint': 1.0,
            'RR_calf_joint': -1.5,

            # Piper arm joints (renamed in combined URDF)
            'arm_joint1': 0.0,
            'arm_joint2': 1.48,
            'arm_joint3': -0.63,
            'arm_joint4': -0.84,
            'arm_joint5': 0.0,
            'arm_joint6': 0.0,
            # Gripper (prismatic)
            'arm_joint7': 0.0,
            'arm_joint8': 0.0,
        }
        rand_yaw_range = np.pi / 2
        origin_perturb_range = 0.5
        init_vel_perturb_range = 0.1

    class control:
        # B2 leg stiffness: higher than B1 due to larger robot
        stiffness = {'joint': 160, 'arm_joint': 5}
        damping = {'joint': 5.0, 'arm_joint': 0.5}

        adaptive_arm_gains = False
        # action scale: 12 leg + 6 arm
        action_scale = [0.4, 0.45, 0.45] * 2 + [0.4, 0.45, 0.45] * 2 + [2.1, 0.6, 0.6, 0, 0, 0]
        decimation = 4
        torque_supervision = False

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/b2piper/urdf/b2_piper.urdf'
        foot_name = "foot"
        gripper_name = "gripper_base"
        penalize_contacts_on = ["thigh", "base_link", "calf"]
        terminate_after_contacts_on = []
        self_collisions = 0  # 1 to disable, 0 to enable
        flip_visual_attachments = False
        collapse_fixed_joints = True
        fix_base_link = False

    class box:
        box_size = 0.1
        randomize_base_mass = True
        added_mass_range = [-0.001, 0.050]
        box_env_origins_x = 0
        box_env_origins_y_range = [0.1, 0.3]
        box_env_origins_z = box_size / 2 + 0.16

    class arm:
        init_target_ee_base = [0.15, 0.0, 0.15]  # EE initial target relative to base (Piper shorter than Z1)
        grasp_offset = 0.08
        osc_kp = np.array([100, 100, 100, 30, 30, 30])
        osc_kd = 2 * (osc_kp ** 0.5)

    class domain_rand:
        observe_priv = True
        randomize_friction = True
        friction_range = [0.3, 3.0]
        randomize_base_mass = True
        added_mass_range = [0., 15.]
        randomize_base_com = True
        added_com_range_x = [-0.15, 0.15]
        added_com_range_y = [-0.15, 0.15]
        added_com_range_z = [-0.15, 0.15]
        randomize_motor = True
        leg_motor_strength_range = [0.7, 1.3]
        arm_motor_strength_range = [0.7, 1.3]
        randomize_gripper_mass = True
        gripper_added_mass_range = [0, 0.1]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5

    class rewards:
        reward_container_name = "maniploco_rewards"

        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_ee_sigma = 1
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 0.4
        base_height_target = 0.53  # B2 target height (slightly crouched with arm)
        max_contact_force = 40.
        gait_vel_sigma = 0.5
        gait_force_sigma = 0.5
        kappa_gait_probs = 0.07
        feet_height_target = 0.3

        feet_aritime_allfeet = False
        feet_height_allfeet = False

        class scales:
            # Gait control rewards
            tracking_contacts_shaped_force = -2.0
            tracking_contacts_shaped_vel = -2.0
            feet_air_time = 2.0
            feet_height = 1.0

            # Tracking rewards
            tracking_lin_vel_max = 2.0
            tracking_lin_vel_x_l1 = 0.
            tracking_lin_vel_x_exp = 0
            tracking_ang_vel = 0.5

            delta_torques = -1.0e-7 / 4.0
            work = 0
            energy_square = 0.0
            torques = -2.5e-5
            stand_still = 1.0
            walking_dof = 1.5
            dof_default_pos = 0.0
            dof_error = 0.0
            alive = 1.0
            lin_vel_z = -1.5
            roll = -2

            # common rewards
            ang_vel_xy = -0.2
            dof_acc = -7.5e-7
            collision = -10.
            action_rate = -0.015
            dof_pos_limits = -10.0
            delta_torques = -1.0e-7
            hip_pos = -0.3
            work = -0.003
            feet_jerk = -0.0002
            feet_drag = -0.08
            feet_contact_forces = -0.001
            orientation = 0.0
            orientation_walking = 0.0
            orientation_standing = 0.0
            base_height = -5.0
            torques_walking = 0.0
            torques_standing = 0.0
            energy_square = 0.0
            energy_square_walking = 0.0
            energy_square_standing = 0.0
            base_height_walking = 0.0
            base_height_standing = 0.0
            penalty_lin_vel_y = 0.

        class arm_scales:
            arm_termination = None
            tracking_ee_sphere = 0.
            tracking_ee_world = 0.8
            tracking_ee_sphere_walking = 0.0
            tracking_ee_sphere_standing = 0.0
            tracking_ee_cart = None
            arm_orientation = None
            arm_energy_abs_sum = None
            tracking_ee_orn = 0.
            tracking_ee_orn_ry = None

    class viewer:
        pos = [-20, 0, 20]
        lookat = [0, 0, -2]

    class termination:
        r_threshold = 0.8
        p_threshold = 0.8
        z_threshold = 0.1

    class terrain:
        mesh_type = 'trimesh'
        hf2mesh_method = "fast"
        max_error = 0.1
        horizontal_scale = 0.05
        vertical_scale = 0.005
        border_size = 25
        height = [0.00, 0.1]
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = False

        all_vertical = False
        no_flat = True

        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20

        terrain_dict = {"smooth slope": 0.,
                        "rough slope up": 0.,
                        "rough slope down": 0.,
                        "rough stairs up": 0.,
                        "rough stairs down": 0.,
                        "discrete": 0.,
                        "stepping stones": 0.,
                        "gaps": 0.,
                        "rough flat": 1.0,
                        "pit": 0.0,
                        "wall": 0.0}
        terrain_proportions = list(terrain_dict.values())
        slope_treshold = None
        origin_zero_z = False


class B2PiperRoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        continue_from_last_std = True
        init_std = [[0.8, 1.0, 1.0] * 4 + [1.0] * 6]
        actor_hidden_dims = [128]
        critic_hidden_dims = [128]
        activation = 'elu'
        output_tanh = False

        leg_control_head_hidden_dims = [128, 128]
        arm_control_head_hidden_dims = [128, 128]

        priv_encoder_dims = [64, 20]

        num_leg_actions = 12
        num_arm_actions = 6

        adaptive_arm_gains = B2PiperRoughCfg.control.adaptive_arm_gains
        adaptive_arm_gains_scale = 10.0

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 2e-4
        schedule = 'fixed'
        gamma = 0.99
        lam = 0.95
        desired_kl = None
        max_grad_norm = 1.
        min_policy_std = [[0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3]

        mixing_schedule = [1.0, 0, 3000]
        torque_supervision = B2PiperRoughCfg.control.torque_supervision
        torque_supervision_schedule = [0.0, 1000, 1000]
        adaptive_arm_gains = B2PiperRoughCfg.control.adaptive_arm_gains
        # dagger params
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 3000, 7000]

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 45000
        save_interval = 200
        experiment_name = 'b2piper_v1'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
