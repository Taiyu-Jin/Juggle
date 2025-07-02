

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import math
import torch
from legged_gym.utils import math
from torch import Tensor
from typing import Tuple, Dict
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .juggle_robot_config import JuggleRobotCfg

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class JuggleRobot(BaseTask):
    def __init__(self, cfg: JuggleRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training
        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.actors_per_env = self.cfg.env.actors_per_env
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.compute_observations()  # 初始化时候先计算一便observation
        self.init_done = True

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        # ------------------这里是载入juggle asset-------------------------------------------
        juggle_asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        juggle_asset_root = os.path.dirname(juggle_asset_path)
        juggle_asset_file = os.path.basename(juggle_asset_path)
        juggle_asset_options = gymapi.AssetOptions()
        juggle_asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        juggle_asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        juggle_asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        juggle_asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        juggle_asset_options.fix_base_link = self.cfg.asset.fix_base_link
        juggle_asset_options.density = self.cfg.asset.density
        juggle_asset_options.angular_damping = self.cfg.asset.angular_damping
        juggle_asset_options.linear_damping = self.cfg.asset.linear_damping
        juggle_asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        juggle_asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        juggle_asset_options.armature = self.cfg.asset.armature
        juggle_asset_options.thickness = self.cfg.asset.thickness
        juggle_asset_options.disable_gravity = self.cfg.asset.disable_gravity
        juggle_asset = self.gym.load_asset(self.sim, juggle_asset_root, juggle_asset_file, juggle_asset_options)
        self.num_dof = self.gym.get_asset_dof_count(juggle_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(juggle_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(juggle_asset)
        juggle_asset_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(juggle_asset)
        #-------------------------------------------------------------------------------------

        # 载入 ball_asset
        ball_radius = 0.1
        mass = 0.4
        volume = (4 / 3) * math.pi * ball_radius ** 3
        density = mass / volume
        ball_options = gymapi.AssetOptions()
        ball_options.density = density
        ball_asset = self.gym.create_sphere(self.sim, ball_radius, ball_options)
        ball_asset_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(ball_asset)



        # save body names from the asset
        juggle_body_names = self.gym.get_asset_rigid_body_names(juggle_asset)
        self.dof_names = self.gym.get_asset_dof_names(juggle_asset)
        self.num_bodies = len(juggle_body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in juggle_body_names if self.cfg.asset.foot_name in s]  # 找到双脚的名字



        # juggle的初始位置    3+4+3+3=13
        juggle_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.juggle_init_state = to_torch(juggle_init_state_list, device=self.device, requires_grad=False)
        ball_init_state_list = self.cfg.ball_init_state.pos + self.cfg.ball_init_state.rot + self.cfg.ball_init_state.lin_vel + self.cfg.ball_init_state.ang_vel
        self.ball_init_state = to_torch(ball_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.juggle_init_state[:3])
        ball_start_pose = gymapi.Transform()
        ball_start_pose.p = gymapi.Vec3(*self.ball_init_state[:3])

        env_spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        self.juggle_handles = []
        self.obj_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create handle
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))


            juggle_rigid_shape_props = self._process_juggle_asset_rigid_shape_props(juggle_asset_rigid_shape_props, i)
            self.gym.set_asset_rigid_shape_properties(juggle_asset, juggle_rigid_shape_props)
            ball_rigid_shape_props = self._process_juggle_asset_rigid_shape_props(ball_asset_rigid_shape_props, i)
            self.gym.set_asset_rigid_shape_properties(ball_asset, ball_rigid_shape_props)

            juggle_handle = self.gym.create_actor(env_handle, juggle_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            self.ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_start_pose, "ball", i, 0, 0)

            # set some properties
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, juggle_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, juggle_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, juggle_handle, body_props, recomputeInertia=True)

            # append handle
            self.envs.append(env_handle)
            self.juggle_handles.append(juggle_handle)
        self.left_foot_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.juggle_handles[0],"L_toe_Link")
        self.right_foot_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.juggle_handles[0], "R_toe_Link")
        self.ball_body_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ball_handle, "sphere")

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        # prepare quantities
        self.ball_pos[:] = self.ball_state[:, 0:3]
        self.ball_lin_vel[:] = self.ball_state[:, 7:10]
        self.right_foot_pos[:] = self.right_foot_state[:, 0:3]
        self.right_foot_quat[:] = self.right_foot_state[:, 3:7]
        self.right_foot_lin_vel[:] = self.right_foot_state[:, 7:10]
        self.right_foot_euler_xyz = get_euler_xyz_tensor(self.right_foot_quat)
        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()   # 从 self.reset_buf 中提取所有需要重置的环境的索引，并将它们展平为一个一维张量 env_ids
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_ball_vel[:] = self.ball_state[:, 7:13]
    def check_termination(self):
        """ Check if environments need to be reset, including ball drop condition.
        """
        # 设置地面高度阈值
        ground_threshold = 0.15
        # 检查球落地和时间超时条件，并更新 reset_buf
        self.reset_buf = self.ball_pos[:, 2] < ground_threshold
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers
        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        # reset dof states
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.
        self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.initial_root_states), gymtorch.unwrap_tensor(multi_env_ids_int32),len(multi_env_ids_int32))

        multi_env_ids_ball_int32 = self._global_indices[env_ids, 1].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.initial_root_states), gymtorch.unwrap_tensor(multi_env_ids_ball_int32), len(multi_env_ids_ball_int32))
        # reset buffers
        self.prev_ball_side[env_ids] = 0
        self.same_side_count[env_ids] = 0
        self.prev_T_mod[env_ids] = 0

        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.right_foot_quat[env_ids] = self.right_foot_state[env_ids, 3:7]
        self.right_foot_euler_xyz = get_euler_xyz_tensor(self.right_foot_quat)

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self): #######                         54
        """ Computes observations
        """
        # print(self.reset_buf)
        T = self._get_T()
        self.compute_ball_and_foot_trajectory()
        self.obs_buf = torch.cat((   self.ball_pos, # 3
                                            self.ball_lin_vel * self.obs_scales.lin_vel,  # 3
                                            self.right_foot_pos,  # 3
                                            self.right_foot_lin_vel * self.obs_scales.lin_vel,  # 3

                                            self.left_foot_pos,  # 3
                                            self.left_foot_lin_vel * self.obs_scales.lin_vel,  # 3

                                           (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 10
                                            self.dof_vel * self.obs_scales.dof_vel,  # 10
                                            self.actions  # 10
                                  ), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,self.sim_params)
        self._create_ground_plane()
        self._create_envs()

    def set_camera(self, position, lookat):
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_juggle_asset_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        # if self.cfg.domain_rand.randomize_friction:   # 刚体摩擦的，先不用管，我在cfg里面设置成false了
        #     if env_id == 0:
        #         # prepare friction randomization
        #         friction_range = self.cfg.domain_rand.friction_range
        #         num_buckets = 64
        #         bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
        #         friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1),
        #                                             device='cpu')
        #         self.friction_coeffs = friction_buckets[bucket_ids]
        #
        #     for s in range(len(props)):
        #         props[s].friction = self.friction_coeffs[env_id]

        # Set restitution to 0.8 for all rigid shapes
        for s in range(len(props)):
            props[s].restitution = 0.6
            props[s].friction = 0.8
            props[s].torsion_friction = 0.8
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item() * self.cfg.safety.pos_limit
                self.dof_pos_limits[i, 1] = props["upper"][i].item() * self.cfg.safety.pos_limit
                self.dof_vel_limits[i] = props["velocity"][i].item() * self.cfg.safety.vel_limit
                self.torque_limits[i] = props["effort"][i].item() * self.cfg.safety.torque_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0.  # previous actions
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, self.actors_per_env, 13)   # 修改root state形状，为了适应两个actor
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)   # 这个简单，就自适应吧 -1代表自适应
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 0, 3:7]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)   # shape: num_envs, num_bodies, xyz axis

        # initial初始化状态
        self.initial_root_states = self.root_states.clone()  # 初始rootstate
        self.initial_root_states[:,1, 7:13] = 0  # 机器人初始 线速度 角速度为0

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.ball_state = self.root_states[:, self.ball_handle, :]
        self.ball_pos = self.ball_state[:, 0:3]
        self.ball_lin_vel = self.ball_state[:, 7:10]
        self.last_ball_vel = torch.zeros_like(self.ball_state[:, 7:13])

        self.right_foot_state = self.rigid_body_state[:, self.right_foot_handle, :]
        self.right_foot_pos = self.right_foot_state[:, 0:3]
        self.right_foot_lin_vel = self.right_foot_state[:, 7:10]
        self.right_foot_quat = self.right_foot_state[:, 3:7]
        self.right_foot_euler_xyz = get_euler_xyz_tensor(self.right_foot_quat)
        self.right_foot_lin_vel = self.right_foot_state[:, 7:10]

        self.left_foot_state = self.rigid_body_state[:, self.left_foot_handle, :]
        self.left_foot_pos = self.left_foot_state[:, 0:3]
        self.left_foot_lin_vel = self.left_foot_state[:, 7:10]
        self.left_foot_quat = self.left_foot_state[:, 3:7]
        self.left_foot_euler_xyz = get_euler_xyz_tensor(self.left_foot_quat)
        self.left_foot_lin_vel = self.left_foot_state[:, 7:10]

        self.distance_ball_foot = torch.norm(self.ball_pos - self.right_foot_pos, dim=-1)
        self._global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        # 初始化追踪球的位置（左侧或右侧）
        self.prev_ball_side = torch.zeros(self.num_envs, device=self.device)
        # 初始化追踪连续在同一侧的周期计数
        self.same_side_count = torch.zeros(self.num_envs, device=self.device)
        # 初始化周期时间，用于检测周期变化
        self.prev_T_mod = torch.zeros(self.num_envs, device=self.device)
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.staticFriction
        plane_params.dynamic_friction = self.cfg.terrain.dynamicFriction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _get_T(self):
        T = self.episode_length_buf * self.dt
        return T

    def compute_ball_and_foot_trajectory(self):
        self.T = self._get_T()

        # ----------------- 球的轨迹 -----------------
        cycle_time_ball = 0.8  # 球的周期为0.8秒
        h0 = 1.3
        h_impact = 0.515
        t_ball = self.T % cycle_time_ball  # 将时间限制在 [0, cycle_time_ball] 范围内

        # 球的下落阶段：从 t = 0 到 t = 0.4
        t_down = t_ball[t_ball <= cycle_time_ball / 2]
        h_down = h0 - (h0 - h_impact) * (t_down / (cycle_time_ball / 2)) ** 2  # 拟合下落轨迹

        # 球的上升阶段：从 t = 0.4 到 t = 0.8
        t_up = t_ball[t_ball > cycle_time_ball / 2] - cycle_time_ball / 2
        h_up = h_impact + (h0 - h_impact) * (t_up / (cycle_time_ball / 2)) ** 2  # 拟合上升轨迹

        # 组合球的高度轨迹
        self.ball_height_trajectory = torch.cat((h_down, h_up))

        # ----------------- 脚的轨迹 -----------------
        cycle_time_foot = 0.8  # 每只脚的周期为1.6秒
        A_z = 0.2  # 振幅为 0.2 米
        h_offset = 0.0  # 初始高度为0
        omega_foot = 2 * torch.pi / cycle_time_foot  # 频率周期为 1.6 秒

        # 设置相位，使得脚在最低点开始运动
        phase_right = -torch.pi / 2  # 右脚从最低点开始
        phase_left = phase_right + torch.pi  # 左脚在相位上滞后180度

        # 根据周期控制右脚和左脚的运动
        foot_phase = (self.T // cycle_time_foot) % 2  # 0 表示右脚运动，1 表示左脚运动


        # 使用 torch.where 来选择脚的高度轨迹
        self.right_foot_height_trajectory = torch.where(
            foot_phase == 0,
            A_z * torch.sin(omega_foot * self.T + phase_right) + h_offset,  # 右脚运动
            torch.tensor(0.0, device=self.device) # 左脚静止
        )

        self.left_foot_height_trajectory = torch.where(
            foot_phase == 1,
            A_z * torch.sin(omega_foot * self.T + phase_left) + h_offset,  # 左脚运动
            torch.tensor(0.0, device=self.device)  # 右脚静止
        )

        # 添加 y 方向的轨迹
        self.left_foot_y_trajectory = A_z * torch.sin(omega_foot * self.T + phase_left)  # Positive y when kicking
        self.right_foot_y_trajectory = -A_z * torch.sin(omega_foot * self.T + phase_right)  # Negative y when kicking

        # ----------------- 球的 Y 方向轨迹 -----------------
        cycle_time_foot = 1.6  # 每只脚的周期为 1.6 秒
        omega_foot = 2 * torch.pi / cycle_time_foot
        omega_ball_y = 2 * omega_foot  # 球的频率是脚的两倍，来回一次的时间是脚踢球的周期

        y_amplitude_ball = 0.2
        T_mod = self.T % cycle_time_foot

        # 分段函数：前半周期球从中间被踢到一侧，后半周期球从一侧回到中间
        t_mod_normalized = (T_mod / cycle_time_foot) * 2  # 归一化周期
        y_trajectory_half_cycle = torch.where(
            T_mod < cycle_time_foot / 2,  # 前半周期：球向一侧移动
            y_amplitude_ball * t_mod_normalized,  # 球线性移动到 y_amplitude_ball
            y_amplitude_ball * (2 - t_mod_normalized)  # 后半周期：球回到中间
        )

        self.ball_y_trajectory = torch.where(
            self.T % (2 * cycle_time_foot) < cycle_time_foot,  # 当前是哪个脚在踢球
            y_trajectory_half_cycle,  # 右脚踢，Y 方向为正
            -y_trajectory_half_cycle  # 左脚踢，Y 方向为负
        )

        # 组合球的完整轨迹（X, Y, Z）
        self.ball_trajectory = torch.stack((
            torch.zeros_like(self.T),  # X direction position, can modify if needed
            self.ball_y_trajectory,
            self.ball_height_trajectory
        ), dim=1)

    # ------------ reward functions----------------

    def _reward_ball_trajectory(self):                     # 奖励球的Z轨迹
        ideal_ball_height = self.ball_height_trajectory
        actual_ball_height = self.ball_pos[:, 2]
        ball_height_diff = actual_ball_height - ideal_ball_height
        ball_reward = torch.exp(-torch.abs(ball_height_diff))
        return ball_reward

    def _reward_ball_height(self):
        """
        控制球的高度，防止球飞得过高。
        """
        # 奖励适中的球高度
        min_height = 0.515
        max_height = 1.3

        # 确保常量的类型和设备一致
        height_reward = torch.where(
            (self.ball_pos[:, 2] >= min_height) & (self.ball_pos[:, 2] <= max_height),
            torch.tensor(1.0, dtype=torch.float, device=self.device),  # 明确为 float 类型
            torch.tensor(0.0, dtype=torch.float, device=self.device)  # 明确为 float 类型
        )

        # 惩罚过高的球高度
        height_penalty = torch.where(
            self.ball_pos[:, 2] > max_height,
            -torch.abs(self.ball_pos[:, 2] - max_height),
            torch.tensor(0.0, dtype=torch.float, device=self.device)  # 确保类型一致
        )

        # 综合奖励
        ball_reward = height_reward + height_penalty
        return ball_reward

    def _reward_foot_height_and_y(self):    #奖励脚往斜上方踢  z和y
        """
        计算脚的高度和y方向位置的奖励。
        左脚应向右上方踢，右脚应向左上方踢，因此不仅需要评估高度差异，还需要评估y方向的位置差异。
        """
        # 理想的脚高度和y位置轨迹
        ideal_right_foot_height = self.right_foot_height_trajectory
        ideal_left_foot_height = self.left_foot_height_trajectory
        ideal_right_foot_y = self.right_foot_y_trajectory
        ideal_left_foot_y = self.left_foot_y_trajectory
        # 实际的脚高度和y位置
        actual_right_foot_height = self.right_foot_pos[:, 2]
        actual_left_foot_height = self.left_foot_pos[:, 2]
        actual_right_foot_y = self.right_foot_pos[:, 1]
        actual_left_foot_y = self.left_foot_pos[:, 1]
        # 计算右脚和左脚的高度差异
        right_foot_height_diff = actual_right_foot_height - ideal_right_foot_height
        left_foot_height_diff = actual_left_foot_height - ideal_left_foot_height
        # 计算右脚和左脚的y位置差异
        right_foot_y_diff = actual_right_foot_y - ideal_right_foot_y
        left_foot_y_diff = actual_left_foot_y - ideal_left_foot_y
        # 计算高度奖励：差异越小，奖励越高
        right_foot_height_reward = torch.exp(-torch.abs(right_foot_height_diff) * 10)  # 调整系数以改变奖励的敏感度
        left_foot_height_reward = torch.exp(-torch.abs(left_foot_height_diff) * 10)
        # 计算y位置奖励：差异越小，奖励越高
        right_foot_y_reward = torch.exp(-torch.abs(right_foot_y_diff) * 10)  # 调整系数以改变奖励的敏感度
        left_foot_y_reward = torch.exp(-torch.abs(left_foot_y_diff) * 10)
        # 组合高度和y位置的奖励
        right_foot_reward = right_foot_height_reward + right_foot_y_reward
        left_foot_reward = left_foot_height_reward + left_foot_y_reward
        # 计算总的脚奖励：左右脚的奖励相加并取平均
        foot_reward = (right_foot_reward + left_foot_reward) / 2.0
        return foot_reward

    def _reward_xy_dist(self): #让脚跟踪球
        # 提取球的 xy 坐标
        ball_x = self.ball_pos[:, 0]
        ball_y = self.ball_pos[:, 1]
        # 提取左右脚的 xy 坐标
        left_foot_x = self.left_foot_pos[:, 0]
        left_foot_y = self.left_foot_pos[:, 1]
        right_foot_x = self.right_foot_pos[:, 0]
        right_foot_y = self.right_foot_pos[:, 1]
        # 脚的偏移量
        instep_left_x = left_foot_x + 0.03
        instep_right_x = right_foot_x + 0.03
        instep_left_y = left_foot_y
        instep_right_y = right_foot_y
        # 计算左右脚与球的距离
        distance_left_x = torch.abs(ball_x - instep_left_x)
        distance_left_y = torch.abs(ball_y - instep_left_y)
        distance_left_xy = torch.sqrt(distance_left_x ** 2 + distance_left_y ** 2)

        distance_right_x = torch.abs(ball_x - instep_right_x)
        distance_right_y = torch.abs(ball_y - instep_right_y)
        distance_right_xy = torch.sqrt(distance_right_x ** 2 + distance_right_y ** 2)
        # 当球 y 坐标小于 -0.2m 时，右脚跟踪球；当球 y 坐标大于 0.2m 时，左脚跟踪球
        distance_xy = torch.where(ball_y < -0.15, distance_right_xy,torch.where(ball_y > 0.15, distance_left_xy,torch.tensor(float('inf'), device=self.device)))
        # 计算奖励，距离越小，奖励越高
        reward = torch.exp(-distance_xy)
        return reward

    def _reward_foot_euler(self):  #让脚面倾斜一个角度，更方便来回踢球
        """
        计算左右脚的欧拉角奖励，鼓励左右脚在踢球时保持一定的倾斜角度，方便左右颠球。
        """
        # 定义左右脚的理想欧拉角度
        # 设定一个左右脚的理想倾斜角度，比如10度（大约为0.1745弧度）
        ideal_right_foot_angle = torch.tensor([0.0, 0.1745], device=self.device)  # 理想的右脚角度，倾斜10度
        ideal_left_foot_angle = torch.tensor([0.0, -0.1745], device=self.device)  # 理想的左脚角度，倾斜-10度
        # 提取左右脚的实际欧拉角（假设欧拉角的维度是 [俯仰角, 侧倾角]）
        actual_right_foot_euler = self.right_foot_euler_xyz[:, :2]  # 提取右脚的俯仰角和侧倾角
        actual_left_foot_euler = self.left_foot_euler_xyz[:, :2]  # 提取左脚的俯仰角和侧倾角
        # 计算右脚和理想角度的差异
        right_foot_angle_diff = torch.abs(actual_right_foot_euler - ideal_right_foot_angle)
        # 计算左脚和理想角度的差异
        left_foot_angle_diff = torch.abs(actual_left_foot_euler - ideal_left_foot_angle)
        # 对角度差异计算指数衰减的奖励，差异越小，奖励越高
        right_foot_angle_reward = torch.exp(-torch.sum(right_foot_angle_diff, dim=1) * 10)  # 调整系数以改变奖励敏感度
        left_foot_angle_reward = torch.exp(-torch.sum(left_foot_angle_diff, dim=1) * 10)
        # 组合左右脚的奖励，取平均值
        quat_mismatch = (right_foot_angle_reward + left_foot_angle_reward) / 2.0
        return quat_mismatch

    def _reward_x_diff(self): #让球远离身体，防止被腿夹住
        # 奖励适中的球高度
        min_x = 0.35
        max_x = 0.25
        #x_reward = torch.where((self.ball_pos[:, 1] >= min_x) & (self.ball_pos[:, 1] <= max_x), 1.0, 0.0)
        x_penalty = torch.where(self.ball_pos[:, 0] < min_x, -torch.abs(self.ball_pos[:, 0] - min_x), torch.tensor(0.0, dtype=self.ball_pos[:, 0].dtype, device=self.device))
        # 综合奖励
        reward = x_penalty
        return reward

    def _reward_ball_y_speed(self):  #奖励球在y方向上的速度，防止球在y；方向上不动
        ball_speed_y = torch.abs(self.ball_lin_vel[:, 1])
        target_speed_y = 0.5
        return torch.exp(-torch.abs(ball_speed_y-target_speed_y))

    def _reward_ball_z_speed(self): #防止球在z方向上停止，没有高度变化
        # 获取球在 Z 方向上的速度
        ball_speed_z = torch.abs(self.ball_lin_vel[:, 2])
        # 对 Z 方向上的速度进行奖励，速度越大奖励越高
        speed_threshold = 1.0
        max_speed = 4.0  # 速度的上限
        # 限制 Z 方向速度在阈值范围内
        clamped_speed_z = torch.clamp(ball_speed_z, min=speed_threshold, max=max_speed)
        # 计算奖励：在阈值范围内，随着速度增加，奖励增加
        motion_reward = torch.exp(clamped_speed_z - speed_threshold)
        return motion_reward

    def _reward_feet_y_pos(self):  #让脚保持在0.2m附近  todo: 这个与跟踪球冲突，好好考虑一下
        # 提取左右脚的 Y 坐标
        left_foot_y = self.left_foot_pos[:, 1]
        right_foot_y = self.right_foot_pos[:, 1]
        # 定义左右脚的目标 Y 位置
        left_foot_target_y = 0.2
        right_foot_target_y = -0.2
        # 计算左脚与目标位置的差异
        left_foot_diff = torch.abs(left_foot_y - left_foot_target_y)
        # 计算右脚与目标位置的差异
        right_foot_diff = torch.abs(right_foot_y - right_foot_target_y)
        # 计算奖励：差异越小，奖励越高
        left_foot_reward = torch.exp(-left_foot_diff * 10)  # 调整 10 来改变奖励的敏感度
        right_foot_reward = torch.exp(-right_foot_diff * 10)
        # 总奖励：左右脚的奖励相加
        total_reward = left_foot_reward + right_foot_reward
        return total_reward

    def _reward_ball_y_pos(self):
        """
        计算球的 Y 位置奖励。
        目标是使球的 Y 位置尽可能接近预定的轨迹，以鼓励交替踢球行为。
        """
        actual_ball_y = self.ball_pos[:, 1]
        # 计算球的 Y 位置与目标位置之间的差异
        y_diff = actual_ball_y - self.ball_y_trajectory
        # 使用指数衰减函数计算奖励，差异越小，奖励越高
        # 调整系数10以控制奖励的敏感度
        reward = torch.exp(-torch.abs(y_diff) * 10)

        return reward

    def _reward_feet_distance(self): # 让两只脚保持；一定的距离，不能靠的太近
        # 提取左右脚的 XYZ 坐标
        left_foot_pos = self.left_foot_pos
        right_foot_pos = self.right_foot_pos
        # 计算两只脚之间的欧几里得距离
        feet_distance = torch.sqrt(torch.sum(torch.square(left_foot_pos - right_foot_pos), dim=1))
        # 定义最小允许的距离，低于该值则给予惩罚
        min_distance = 0.35  # 设置为 0.3 米
        # 惩罚机制：如果距离小于阈值，给予惩罚
        distance_penalty = torch.where(feet_distance < min_distance,
                                       torch.exp(min_distance - feet_distance),  # 距离越小惩罚越大
                                       torch.tensor(0.0, device=self.device))  # 超过最小距离无惩罚
        return -distance_penalty  # 返回负的惩罚

    def _reward_same_side(self):
        # 定义周期
        cycle_time = 0.8  # 每个周期为 0.8s
        T_mod = (self.T % cycle_time)  # 当前时间在周期中的位置
        # 获取球的 Y 位置
        ball_y = self.ball_pos[:, 1]
        # 当前周期内球的位置
        current_side = torch.where(ball_y > 0, torch.tensor(1.0, device=self.device),
                                   torch.tensor(0.0, device=self.device))
        # 判断是否进入新周期，并明确类型为布尔类型
        new_cycle = (T_mod < self.prev_T_mod).bool()  # 当 T_mod 小于之前的 T_mod 时，表示进入了新周期
        # 判断球是否保持在同一侧
        same_side = (current_side == self.prev_ball_side).bool()
        # 更新计数和状态，仅当进入新周期且在同一侧时才更新 same_side_count
        self.same_side_count = torch.where(new_cycle & same_side, self.same_side_count + 1,
                                           torch.zeros_like(self.same_side_count))
        # 更新上一个周期球的位置，仅当进入新周期时更新
        self.prev_ball_side = torch.where(new_cycle, current_side, self.prev_ball_side)
        # 记录当前的 T_mod 作为下一步的 prev_T_mod
        self.prev_T_mod = T_mod
        # 计算逐渐增加的惩罚，随着 same_side_count 的增加惩罚加重
        penalty = torch.exp(self.same_side_count * 1.0)  # 增大系数以加强惩罚
        # 限制惩罚的最大值，防止过大
        max_penalty = 100.0
        penalty = torch.clamp(penalty, max=max_penalty)
        # 返回总的惩罚，惩罚值为负
        return -penalty

    def _reward_alternation(self):
        # 定义周期
        cycle_time = 0.8  # 每个周期为 0.8s
        T_mod = (self.T % cycle_time)  # 当前时间在周期中的位置

        # 获取球的 Y 位置
        ball_y = self.ball_pos[:, 1]

        # 当前周期内球的位置
        current_side = torch.where(ball_y > 0, torch.tensor(1.0, device=self.device),
                                   torch.tensor(0.0, device=self.device))

        # 判断是否进入新周期，并明确类型为布尔类型
        new_cycle = (T_mod < self.prev_T_mod).bool()  # 当 T_mod 小于之前的 T_mod 时，表示进入了新周期

        # 判断球是否保持在同一侧
        same_side = (current_side == self.prev_ball_side).bool()

        # 初始化奖励张量
        alternation_reward = torch.zeros(self.num_envs, device=self.device)

        # 当进入新周期且球切换到另一侧时，给予奖励
        alternation_reward = torch.where(new_cycle & ~same_side, torch.tensor(1.0, device=self.device),
                                         alternation_reward)

        # 更新上一个周期球的位置，仅当进入新周期时更新
        self.prev_ball_side = torch.where(new_cycle, current_side, self.prev_ball_side)

        # 记录当前的 T_mod 作为下一步的 prev_T_mod
        self.prev_T_mod = T_mod

        return alternation_reward

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_action_smoothness(self):
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3