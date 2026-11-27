from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_matrix

from . import geometry as geo

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(SIM_DIR, "assets", "twolink", "twolinkbot.urdf")
USD_DIR = os.path.join(SIM_DIR, "assets", "usd")

JOINT_STIFFNESS = {"joint1": 4.0, "joint2": 1.0}
JOINT_DAMPING = {"joint1": 0.4, "joint2": 0.08}

TWOLINK_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        usd_dir=USD_DIR,
        fix_base=True,
        merge_fixed_joints=True,
        make_instanceable=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=JOINT_STIFFNESS, damping=JOINT_DAMPING
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={"joint1": 0.0, "joint2": math.pi / 2}),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint1", "joint2"],
            stiffness=JOINT_STIFFNESS,
            damping=JOINT_DAMPING,
            effort_limit_sim=10.0,
            velocity_limit_sim=3.0,
        )
    },
)


@configclass
class MarkerTrackEnvCfg(DirectRLEnvCfg):
    decimation = 10
    episode_length_s = 20.0
    action_space = 2
    observation_space = 10
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation)
    robot_cfg: ArticulationCfg = TWOLINK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=1.5, replicate_physics=True, clone_in_fabric=False
    )
    show_marker = False

    d_goal = 0.20
    q_lower = (-3.0, -3.0)
    q_upper = (3.0, 3.0)
    max_dq = 0.2
    joint_v_max = 2.0
    start_pose = (0.0, math.pi / 2)
    start_noise = 0.1
    random_start_prob = 0.5
    random_start_frac = 0.8

    command_delay_steps = (1, 2)
    lag_tau_range = ((0.15, 0.25), (0.20, 0.50))
    deadband_range = ((0.003, 0.005), (0.0, 0.005))
    stick_prob_range = (0.0, 0.05)
    stick_steps = (2, 8)
    obs_latency_steps = 1
    stale_obs_prob = 0.05

    mount_pos_nominal = (0.10853, 0.03233, 0.02864)
    mount_pos_range = (0.01, 0.01, 0.005)
    mount_yaw_range = math.radians(5.0)

    hfov_tan_half = 324.964 / 606.542
    vfov_tan_half = 242.520 / 606.322
    z_range = (0.08, 1.5)
    max_view_angle = math.radians(70.0)
    marker_half_size = 0.03

    q_quant = 2 * math.pi / 16384
    q_noise = 0.001
    pos_noise = 0.002
    z_noise_frac = 0.005
    z_bias_range = (0.995, 1.005)
    psi_flip_angle = math.radians(8.0)
    psi_flip_prob = 0.5
    psi_gain_range = (0.6, 1.0)
    psi_noise_range = (math.radians(1.0), math.radians(5.0))
    dropout_prob_range = (0.0, 0.1)

    marker_waypoint_q_frac = 0.95
    marker_min_radius = 0.15
    marker_dist_range = (0.10, 0.55)
    marker_lateral_frac = 0.6
    marker_height_range = (-0.04, 0.04)
    marker_yaw_offset_range = math.radians(35.0)
    marker_speed_range = (0.02, 0.30)
    marker_hold_prob = 0.5
    marker_hold_range = (0.0, 3.0)
    marker_static_prob = 0.15
    marker_in_view_prob = 0.6
    marker_jitter_prob = 0.5
    marker_jitter_amp = 0.02
    marker_jitter_freq_range = (0.4, 1.5)

    w_center = 1.0
    w_dist = 1.0
    w_parallel = 0.3
    sigma_center = 0.1
    sigma_dist = 0.05
    sigma_parallel = 0.2
    w_lost = -1.0
    w_dist_lin = -2.0
    w_center_lin = -0.5
    w_action = -0.01
    w_action_rate = -0.1
    w_limit = -0.5
    limit_margin_frac = 0.95


class MarkerTrackEnv(DirectRLEnv):
    cfg: MarkerTrackEnvCfg

    def __init__(self, cfg: MarkerTrackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self._joint_ids, _ = self.robot.find_joints(["joint1", "joint2"], preserve_order=True)
        self.q_lower = torch.tensor(self.cfg.q_lower, device=dev)
        self.q_upper = torch.tensor(self.cfg.q_upper, device=dev)

        self.q_cmd = torch.zeros(n, 2, device=dev)
        max_delay = max(self.cfg.command_delay_steps)
        self.cmd_queue = torch.zeros(n, max_delay + 1, 2, device=dev)
        self.cmd_delay = torch.zeros(n, dtype=torch.long, device=dev)
        self.q_applied = torch.zeros(n, 2, device=dev)
        self.q_ref = torch.zeros(n, 2, device=dev)
        self.lag_tau = torch.ones(n, 2, device=dev)
        self.deadband = torch.zeros(n, 2, device=dev)
        self.stick_prob = torch.zeros(n, device=dev)
        self.stick_left = torch.zeros(n, dtype=torch.long, device=dev)
        self.actions = torch.zeros(n, 2, device=dev)
        self.prev_actions = torch.zeros(n, 2, device=dev)

        self.mount_pos = torch.zeros(n, 3, device=dev)
        self.mount_yaw = torch.zeros(n, device=dev)
        self.q_offset = torch.zeros(n, 2, device=dev)
        self.z_bias = torch.ones(n, device=dev)
        self.psi_gain = torch.ones(n, device=dev)
        self.psi_noise = torch.zeros(n, device=dev)
        self.dropout_prob = torch.zeros(n, device=dev)

        self.mk_time = torch.zeros(n, device=dev)
        self.mk_static = torch.zeros(n, dtype=torch.bool, device=dev)
        self.mk_base = torch.zeros(n, 3, device=dev)
        self.mk_yaw = torch.zeros(n, device=dev)
        self.mk_wp = torch.zeros(n, 3, device=dev)
        self.mk_wp_yaw = torch.zeros(n, device=dev)
        self.mk_speed = torch.zeros(n, device=dev)
        self.mk_hold = torch.zeros(n, device=dev)
        self.mk_jit_amp = torch.zeros(n, device=dev)
        self.mk_jit_freq = torch.zeros(n, device=dev)
        self.mk_jit_phase = torch.zeros(n, device=dev)
        self.mk_jit_dir = torch.zeros(n, 3, device=dev)
        self.marker_pos = torch.zeros(n, 3, device=dev)
        self.marker_rot = torch.eye(3, device=dev).repeat(n, 1, 1)

        self.p_cam = torch.zeros(n, 3, device=dev)
        self.r_cam = torch.eye(3, device=dev).repeat(n, 1, 1)
        self.e_theta = torch.zeros(n, device=dev)
        self.e_d = torch.zeros(n, device=dev)
        self.e_psi = torch.zeros(n, device=dev)
        self.is_visible = torch.zeros(n, dtype=torch.bool, device=dev)

        lat = self.cfg.obs_latency_steps
        self.meas_queue = torch.zeros(n, lat + 1, 5, device=dev)
        self.last_seen = torch.zeros(n, 4, device=dev)
        self.steps_unseen = torch.zeros(n, device=dev)
        self.last_obs = torch.zeros(n, self.cfg.observation_space, device=dev)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.marker_vis = None
        if self.cfg.show_marker:
            self.marker_vis = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/ArucoMarker",
                    markers={
                        "marker": sim_utils.CuboidCfg(
                            size=(0.06, 0.06, 0.004),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
                        )
                    },
                )
            )

    def _uniform(self, lo, hi, shape) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(shape, device=self.device)

    def _sample_waypoint(self, env_ids, q=None):
        c, k = self.cfg, len(env_ids)
        if q is None:
            q = self._uniform(self.q_lower * c.marker_waypoint_q_frac, self.q_upper * c.marker_waypoint_q_frac, (k, 2))
        cam_pos, cam_rot = geo.camera_pose(q, self.mount_pos[env_ids], self.mount_yaw[env_ids])
        dist = self._uniform(*c.marker_dist_range, (k,))
        lateral = self._uniform(-c.marker_lateral_frac, c.marker_lateral_frac, (k,)) * c.hfov_tan_half * dist
        pos = cam_pos + dist.unsqueeze(-1) * cam_rot[:, :, 2] + lateral.unsqueeze(-1) * cam_rot[:, :, 0]
        pos[:, 2] = cam_pos[:, 2] + self._uniform(*c.marker_height_range, (k,))
        cam_yaw = torch.atan2(cam_rot[:, 1, 2], cam_rot[:, 0, 2])
        yaw = cam_yaw + math.pi + self._uniform(-c.marker_yaw_offset_range, c.marker_yaw_offset_range, (k,))
        return pos, torch.atan2(torch.sin(yaw), torch.cos(yaw))

    def _update_marker(self, advance=True):
        c, n, dev = self.cfg, self.num_envs, self.device
        if advance:
            dt = self.step_dt
            holding = self.mk_hold > 0
            self.mk_hold = (self.mk_hold - dt).clamp(min=0.0)
            moving = ~holding & ~self.mk_static
            delta = self.mk_wp - self.mk_base
            dist = delta.norm(dim=-1)
            step = torch.minimum(self.mk_speed * dt, dist)
            frac = torch.where(moving, step / dist.clamp(min=1e-6), torch.zeros_like(dist))
            self.mk_base = self.mk_base + frac.unsqueeze(-1) * delta
            dyaw = torch.atan2(torch.sin(self.mk_wp_yaw - self.mk_yaw), torch.cos(self.mk_wp_yaw - self.mk_yaw))
            self.mk_yaw = self.mk_yaw + frac * dyaw
            arrived = (moving & (dist - step < 1e-4)).nonzero(as_tuple=False).squeeze(-1)
            if len(arrived) > 0:
                k = len(arrived)
                wp, wp_yaw = self._sample_waypoint(arrived)
                self.mk_wp[arrived] = wp
                self.mk_wp_yaw[arrived] = wp_yaw
                self.mk_speed[arrived] = self._uniform(*c.marker_speed_range, (k,))
                hold = self._uniform(*c.marker_hold_range, (k,))
                self.mk_hold[arrived] = torch.where(torch.rand(k, device=dev) < c.marker_hold_prob, hold, torch.zeros_like(hold))
        jitter = self.mk_jit_amp * torch.sin(2 * math.pi * self.mk_jit_freq * self.mk_time + self.mk_jit_phase)
        pos = self.mk_base + jitter.unsqueeze(-1) * self.mk_jit_dir
        r_xy = pos[:, :2].norm(dim=-1, keepdim=True)
        pos[:, :2] = pos[:, :2] * (c.marker_min_radius / r_xy.clamp(min=1e-6)).clamp(min=1.0)
        self.marker_pos = pos
        self.marker_rot = geo.marker_rotation(self.mk_yaw)

    def _compute_state(self):
        q = self.robot.data.joint_pos[:, self._joint_ids]
        cam_pos, cam_rot = geo.camera_pose(q, self.mount_pos, self.mount_yaw)
        self.p_cam, self.r_cam = geo.marker_in_camera(cam_pos, cam_rot, self.marker_pos, self.marker_rot)
        self.e_theta, self.e_d, self.e_psi = geo.tracking_errors(self.p_cam, self.r_cam, self.cfg.d_goal)
        self.is_visible = geo.visible(
            self.p_cam, self.r_cam, self.cfg.hfov_tan_half, self.cfg.vfov_tan_half, self.cfg.z_range,
            self.cfg.max_view_angle, self.cfg.marker_half_size,
        )

    def _measure(self) -> torch.Tensor:
        n = self.num_envs
        pos = self.p_cam + self.cfg.pos_noise * torch.randn(n, 3, device=self.device)
        z = self.p_cam[:, 2]
        pos[:, 2] = z * self.z_bias + self.cfg.z_noise_frac * z * torch.randn(n, device=self.device)
        psi = self.psi_gain * self.e_psi + self.psi_noise * torch.randn(n, device=self.device)
        flip = (self.e_psi.abs() < self.cfg.psi_flip_angle) & (torch.rand(n, device=self.device) < self.cfg.psi_flip_prob)
        psi = torch.where(flip, -psi, psi)
        seen = self.is_visible & (torch.rand(n, device=self.device) >= self.dropout_prob)
        return torch.cat((pos, psi.unsqueeze(-1), seen.float().unsqueeze(-1)), dim=-1)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.q_cmd = torch.max(torch.min(self.q_cmd + self.cfg.max_dq * self.actions, self.q_upper), self.q_lower)
        self.cmd_queue = torch.roll(self.cmd_queue, 1, dims=1)
        self.cmd_queue[:, 0] = self.q_cmd
        self.q_applied = self.cmd_queue[torch.arange(self.num_envs, device=self.device), self.cmd_delay]
        moving2 = (self.q_applied[:, 1] - self.q_ref[:, 1]).abs() > self.deadband[:, 1]
        start = (self.stick_left == 0) & moving2 & (torch.rand(self.num_envs, device=self.device) < self.stick_prob)
        n_stick = torch.randint(self.cfg.stick_steps[0], self.cfg.stick_steps[1] + 1, (self.num_envs,), device=self.device)
        self.stick_left = torch.where(start, n_stick, (self.stick_left - 1).clamp(min=0))
        self.mk_time += self.step_dt
        self._update_marker(advance=True)

    def _apply_action(self) -> None:
        err = self.q_applied - self.q_ref
        alpha = (self.physics_dt / self.lag_tau).clamp(max=1.0)
        move = err.abs() > self.deadband
        move[:, 1] &= self.stick_left == 0
        v_step = self.cfg.joint_v_max * self.physics_dt
        self.q_ref = self.q_ref + torch.where(move, (alpha * err).clamp(-v_step, v_step), torch.zeros_like(err))
        self.robot.set_joint_position_target(self.q_ref, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict:
        self._compute_state()
        meas = self._measure()
        self.meas_queue = torch.roll(self.meas_queue, 1, dims=1)
        self.meas_queue[:, 0] = meas
        m = self.meas_queue[:, self.cfg.obs_latency_steps]
        seen = m[:, 4] > 0.5
        self.last_seen = torch.where(seen.unsqueeze(-1), m[:, :4], self.last_seen)
        self.steps_unseen = torch.where(seen, torch.zeros_like(self.steps_unseen), self.steps_unseen + 1)
        q = self.robot.data.joint_pos[:, self._joint_ids] + self.q_offset
        q = q + self.cfg.q_noise * torch.randn_like(q)
        q = torch.round(q / self.cfg.q_quant) * self.cfg.q_quant
        obs = torch.cat(
            (
                self.last_seen[:, 0:2],
                (self.last_seen[:, 2] - self.cfg.d_goal).unsqueeze(-1),
                self.last_seen[:, 3:4],
                seen.float().unsqueeze(-1),
                (self.steps_unseen / 10.0).clamp(max=1.0).unsqueeze(-1),
                q,
                self.q_cmd,
            ),
            dim=-1,
        )
        stale = (torch.rand(self.num_envs, device=self.device) < self.cfg.stale_obs_prob) & (self.episode_length_buf > 0)
        obs = torch.where(stale.unsqueeze(-1), self.last_obs, obs)
        self.last_obs = obs.clone()
        if self.marker_vis is not None:
            self.marker_vis.visualize(
                translations=self.marker_pos + self.scene.env_origins, orientations=quat_from_matrix(self.marker_rot)
            )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._compute_state()
        vis = self.is_visible.float()
        r_center = torch.exp(-torch.square(self.e_theta / self.cfg.sigma_center))
        r_dist = torch.exp(-torch.square(self.e_d / self.cfg.sigma_dist))
        r_par = torch.exp(-torch.square(self.e_psi / self.cfg.sigma_parallel))
        r_track = vis * (self.cfg.w_center * r_center + self.cfg.w_dist * r_dist + self.cfg.w_parallel * r_par)
        r_lin = vis * (self.cfg.w_dist_lin * self.e_d.abs() + self.cfg.w_center_lin * self.e_theta.abs())
        r_lost = self.cfg.w_lost * (1.0 - vis)
        r_act = self.cfg.w_action * torch.sum(torch.square(self.actions), dim=-1)
        r_rate = self.cfg.w_action_rate * torch.sum(torch.square(self.actions - self.prev_actions), dim=-1)
        q = self.robot.data.joint_pos[:, self._joint_ids]
        frac = torch.where(q >= 0, q / self.q_upper, q / self.q_lower)
        r_limit = self.cfg.w_limit * torch.sum((frac - self.cfg.limit_margin_frac).clamp(min=0.0) / (1 - self.cfg.limit_margin_frac), dim=-1)
        total = r_track + r_lin + r_lost + r_act + r_rate + r_limit
        log = self.extras.setdefault("log", {})
        log["Episode_Reward/center"] = (vis * r_center).mean()
        log["Episode_Reward/dist"] = (vis * r_dist).mean()
        log["Episode_Reward/parallel"] = (vis * r_par).mean()
        log["Episode_Reward/linear_err"] = r_lin.mean()
        log["Metrics/visible"] = vis.mean()
        n_vis = vis.sum().clamp(min=1.0)
        log["Metrics/abs_e_theta_deg"] = (vis * torch.rad2deg(self.e_theta.abs())).sum() / n_vis
        log["Metrics/abs_e_d_mm"] = (vis * 1000 * self.e_d.abs()).sum() / n_vis
        log["Metrics/abs_e_psi_deg"] = (vis * torch.rad2deg(self.e_psi.abs())).sum() / n_vis
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        k, dev, c = len(env_ids), self.device, self.cfg

        q = torch.tensor(c.start_pose, device=dev).repeat(k, 1) + self._uniform(-c.start_noise, c.start_noise, (k, 2))
        rand_q = self._uniform(self.q_lower * c.random_start_frac, self.q_upper * c.random_start_frac, (k, 2))
        use_rand = torch.rand(k, device=dev) < c.random_start_prob
        q = torch.where(use_rand.unsqueeze(-1), rand_q, q)
        q = torch.max(torch.min(q, self.q_upper), self.q_lower)
        zeros = torch.zeros_like(q)
        self.robot.write_joint_state_to_sim(q, zeros, self._joint_ids, env_ids)
        self.robot.set_joint_position_target(q, joint_ids=self._joint_ids, env_ids=env_ids)
        self.q_cmd[env_ids] = q
        self.q_applied[env_ids] = q
        self.q_ref[env_ids] = q
        lag = torch.tensor(c.lag_tau_range, device=dev)
        self.lag_tau[env_ids] = lag[:, 0] + (lag[:, 1] - lag[:, 0]) * torch.rand(k, 2, device=dev)
        db = torch.tensor(c.deadband_range, device=dev)
        self.deadband[env_ids] = db[:, 0] + (db[:, 1] - db[:, 0]) * torch.rand(k, 2, device=dev)
        self.stick_prob[env_ids] = self._uniform(*c.stick_prob_range, (k,))
        self.stick_left[env_ids] = 0
        self.cmd_queue[env_ids] = q.unsqueeze(1)
        self.cmd_delay[env_ids] = torch.randint(c.command_delay_steps[0], c.command_delay_steps[1] + 1, (k,), device=dev)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

        nominal = torch.tensor(c.mount_pos_nominal, device=dev)
        span = torch.tensor(c.mount_pos_range, device=dev)
        self.mount_pos[env_ids] = nominal + self._uniform(-span, span, (k, 3))
        self.mount_yaw[env_ids] = self._uniform(-c.mount_yaw_range, c.mount_yaw_range, (k,))
        self.q_offset[env_ids] = self._uniform(-0.003, 0.003, (k, 2))
        self.z_bias[env_ids] = self._uniform(*c.z_bias_range, (k,))
        self.psi_gain[env_ids] = self._uniform(*c.psi_gain_range, (k,))
        self.psi_noise[env_ids] = self._uniform(*c.psi_noise_range, (k,))
        self.dropout_prob[env_ids] = self._uniform(*c.dropout_prob_range, (k,))

        self.mk_time[env_ids] = 0.0
        self.mk_static[env_ids] = torch.rand(k, device=dev) < c.marker_static_prob
        start_view, yaw_view = self._sample_waypoint(env_ids, q)
        start_rand, yaw_rand = self._sample_waypoint(env_ids)
        in_view = (torch.rand(k, device=dev) < c.marker_in_view_prob).unsqueeze(-1)
        self.mk_base[env_ids] = torch.where(in_view, start_view, start_rand)
        self.mk_yaw[env_ids] = torch.where(in_view.squeeze(-1), yaw_view, yaw_rand)
        wp, wp_yaw = self._sample_waypoint(env_ids)
        self.mk_wp[env_ids] = wp
        self.mk_wp_yaw[env_ids] = wp_yaw
        self.mk_speed[env_ids] = self._uniform(*c.marker_speed_range, (k,))
        self.mk_hold[env_ids] = self._uniform(0.0, 2.0, (k,))
        jit_on = torch.rand(k, device=dev) < c.marker_jitter_prob
        self.mk_jit_amp[env_ids] = torch.where(jit_on, self._uniform(0.0, c.marker_jitter_amp, (k,)), torch.zeros(k, device=dev))
        self.mk_jit_freq[env_ids] = self._uniform(*c.marker_jitter_freq_range, (k,))
        self.mk_jit_phase[env_ids] = self._uniform(0.0, 2 * math.pi, (k,))
        jd = torch.randn(k, 3, device=dev)
        jd[:, 2] *= 0.3
        self.mk_jit_dir[env_ids] = jd / jd.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self._update_marker(advance=False)

        self.meas_queue[env_ids] = 0.0
        self.last_seen[env_ids] = 0.0
        self.steps_unseen[env_ids] = 0.0
        self.last_obs[env_ids] = 0.0
