import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from isaacgym import gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.manip_loco.b2piper_config import B2PiperRoughCfg


def parse_args():
    return gymutil.parse_arguments(
        description="Load and visualize b2piper URDF directly in Isaac Gym.",
        custom_parameters=[
            {
                "name": "--asset_file",
                "type": str,
                "default": "resources/robots/b2piper/urdf/b2_piper.urdf",
                "help": "Asset path relative to LEGGED_GYM_ROOT_DIR.",
            },
            {"name": "--fix_base_link", "type": int, "default": 1, "help": "1 fixes the base link."},
            {"name": "--disable_gravity", "type": int, "default": 1, "help": "1 disables gravity on the asset."},
            {"name": "--collapse_fixed_joints", "type": int, "default": 1, "help": "1 collapses fixed joints."},
            {"name": "--flip_visual_attachments", "type": int, "default": 0, "help": "1 flips visual attachments."},
            {"name": "--replace_cylinder_with_capsule", "type": int, "default": 1, "help": "1 replaces collision cylinders with capsules."},
            {"name": "--self_collisions", "type": int, "default": 0, "help": "Actor self-collision flag."},
            {"name": "--base_height", "type": float, "default": 0.58, "help": "Initial base height."},
            {"name": "--stiffness", "type": float, "default": 400.0, "help": "Position drive stiffness for all DOFs."},
            {"name": "--damping", "type": float, "default": 40.0, "help": "Position drive damping for all DOFs."},

        ],
    )


def make_sim(gym, args):
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
    else:
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym sim.")
    return sim


def add_ground(gym, sim):
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0.0
    gym.add_ground(sim, plane_params)


def load_b2piper_asset(gym, sim, args):
    asset_path = os.path.join(LEGGED_GYM_ROOT_DIR, args.asset_file)
    asset_root = os.path.dirname(asset_path)
    asset_file = os.path.basename(asset_path)

    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.fix_base_link = bool(args.fix_base_link)
    asset_options.disable_gravity = bool(args.disable_gravity)
    asset_options.collapse_fixed_joints = bool(args.collapse_fixed_joints)
    asset_options.flip_visual_attachments = bool(args.flip_visual_attachments)
    asset_options.replace_cylinder_with_capsule = bool(args.replace_cylinder_with_capsule)
    asset_options.use_mesh_materials = True
    asset_options.density = 0.001
    asset_options.angular_damping = 0.0
    asset_options.linear_damping = 0.0
    asset_options.max_angular_velocity = 1000.0
    asset_options.max_linear_velocity = 1000.0
    asset_options.armature = 0.0
    asset_options.thickness = 0.01

    print("Loading asset:")
    print(f"  asset_root = {asset_root}")
    print(f"  asset_file = {asset_file}")
    print("Asset options:")
    print(f"  fix_base_link = {asset_options.fix_base_link}")
    print(f"  disable_gravity = {asset_options.disable_gravity}")
    print(f"  collapse_fixed_joints = {asset_options.collapse_fixed_joints}")
    print(f"  flip_visual_attachments = {asset_options.flip_visual_attachments}")
    print(f"  replace_cylinder_with_capsule = {asset_options.replace_cylinder_with_capsule}")

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    if asset is None:
        raise RuntimeError(f"Failed to load asset: {asset_path}")
    return asset


def print_asset_info(gym, asset):
    body_names = gym.get_asset_rigid_body_names(asset)
    dof_names = gym.get_asset_dof_names(asset)
    dof_dict = gym.get_asset_dof_dict(asset)
    body_dict = gym.get_asset_rigid_body_dict(asset)

    print("\nAsset summary:")
    print(f"  num_bodies = {len(body_names)}")
    print(f"  num_dofs = {len(dof_names)}")
    print("\nDOF order:")
    for i, name in enumerate(dof_names):
        print(f"  {i:02d}: {name}")
    print("\nRigid body order:")
    for i, name in enumerate(body_names):
        print(f"  {i:02d}: {name}")
    print("\nDOF dict:")
    print(sorted(dof_dict.items(), key=lambda item: item[1]))
    print("\nBody dict:")
    print(sorted(body_dict.items(), key=lambda item: item[1]))


def configure_dofs(gym, asset, env, actor, stiffness, damping):
    dof_names = gym.get_asset_dof_names(asset)
    dof_props = gym.get_asset_dof_properties(asset)
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(stiffness)
    dof_props["damping"].fill(damping)
    gym.set_actor_dof_properties(env, actor, dof_props)

    default_angles = B2PiperRoughCfg.init_state.default_joint_angles
    default_pos = np.zeros(len(dof_names), dtype=np.float32)
    for i, name in enumerate(dof_names):
        default_pos[i] = default_angles.get(name, 0.0)

    dof_states = np.zeros(len(dof_names), dtype=gymapi.DofState.dtype)
    dof_states["pos"] = default_pos
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, actor, default_pos)

    print("\nDefault DOF pose:")
    for i, name in enumerate(dof_names):
        print(f"  {i:02d}: {name:20s} {default_pos[i]: .4f}")


def create_env_and_actor(gym, sim, asset, args):
    env_lower = gymapi.Vec3(-1.5, -1.5, 0.0)
    env_upper = gymapi.Vec3(1.5, 1.5, 1.5)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, args.base_height)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor = gym.create_actor(env, asset, pose, "b2piper", 0, args.self_collisions, 0)
    configure_dofs(gym, asset, env, actor, args.stiffness, args.damping)
    return env, actor


def main():
    args = parse_args()
    gym = gymapi.acquire_gym()
    sim = make_sim(gym, args)
    add_ground(gym, sim)

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        gym.destroy_sim(sim)
        raise RuntimeError("Failed to create viewer. Check graphics device / DISPLAY.")

    asset = load_b2piper_asset(gym, sim, args)
    print_asset_info(gym, asset)
    create_env_and_actor(gym, sim, asset, args)

    gym.viewer_camera_look_at(

        viewer,
        None,
        gymapi.Vec3(2.2, -2.2, 1.2),
        gymapi.Vec3(0.0, 0.0, 0.35),
    )

    print("\nViewer opened. Close the viewer window to exit.")
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
