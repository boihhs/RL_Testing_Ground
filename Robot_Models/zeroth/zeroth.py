import jax.numpy as jnp, jax
from jax import random
from jax import lax
from jax.tree_util import register_pytree_node_class
from functools import partial
# from Mujoco_Env.Sim import Sim
import optax
from flax.training import checkpoints
from pathlib import Path
from dataclasses import dataclass, field
import yaml
import matplotlib.pyplot as plt
from jax import debug
from mujoco import mjx
import mujoco
from flax.core import FrozenDict

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Sensor:
    id: str     = field(metadata={'static': True})
    start: int     = field(metadata={'static': True})
    length: int     = field(metadata={'static': True})


@jax.jit
def get_obs_and_reward_walking(env, sim, key):

    def _quat_to_small_euler(q):
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        yaw   = jnp.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        pitch = jnp.arcsin(jnp.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
        roll  = jnp.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        return roll, pitch, yaw
    
    def _rotate_vector_inverse_rpy(roll, pitch, yaw, vector):
        R_x = jnp.array([[1, 0, 0], [0, jnp.cos(roll), -jnp.sin(roll)], [0, jnp.sin(roll), jnp.cos(roll)]])
        R_y = jnp.array([[jnp.cos(pitch), 0, jnp.sin(pitch)], [0, 1, 0], [-jnp.sin(pitch), 0, jnp.cos(pitch)]])
        R_z = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0], [jnp.sin(yaw), jnp.cos(yaw), 0], [0, 0, 1]])
        return (R_z @ R_y @ R_x).T @ vector

    # ---------------- Core state & sensors ----------------
    d = env.mjx_data
    step_num = env.step_num
    current_action = env.curr_action
    prev_action = env.prev_action
    goal_velocity = env.goal_velocity           # (vx_cmd, vy_cmd, wz_cmd) in BODY frame
    push_force = env.force_applied
    body_mass = jnp.sum(sim.mjx_model.body_mass)

    ang_vel_id = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_SENSOR, "global_angvel")
    ang_vel_sensor = Sensor(ang_vel_id, sim.mjx_model.sensor_adr[ang_vel_id], sim.mjx_model.sensor_dim[ang_vel_id])

    lin_accel_id = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer")
    lin_accel_sensor = Sensor(lin_accel_id, sim.mjx_model.sensor_adr[lin_accel_id], sim.mjx_model.sensor_dim[lin_accel_id])

    right_foot_body_id = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_BODY, "foot_right")
    left_foot_body_id  = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_BODY, "foot_left")
    body_id = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_BODY, "base")

    right_foot_geom_id = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_GEOM, "foot_right_collision")
    left_foot_geom_id  = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_GEOM, "foot_left_collision")
    ground_geom_id     = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    body_pos   = d.qpos[:3]
    body_q     = d.qpos[3:7]
    body_vel_w = d.qvel[:3]         # world frame; rotate to body below
    joint_pos  = d.qpos[7:]
    joint_vel  = d.qvel[6:]
    joint_accel= d.qacc[6:]

    base_ang_vel = d.sensordata[ang_vel_sensor.start:ang_vel_sensor.start + ang_vel_sensor.length]  # world angular vel
    base_lin_accel = d.sensordata[lin_accel_sensor.start:lin_accel_sensor.start + lin_accel_sensor.length]
    base_ang_accel = d.qacc[3:6]

    right_foot_pos = d.xpos[right_foot_body_id]
    right_foot_vel = d.cvel[right_foot_body_id][3:]
    right_foot_q   = d.xquat[right_foot_body_id]
    right_foot_ang_vel = d.cvel[right_foot_body_id][:3]
    right_foot_force = d._impl.cfrc_ext[right_foot_body_id][3:]
    right_foot_ground_contact = sim.get_collision(d, right_foot_geom_id, ground_geom_id)

    left_foot_pos = d.xpos[left_foot_body_id]
    left_foot_vel = d.cvel[left_foot_body_id][3:]
    left_foot_q   = d.xquat[left_foot_body_id]
    left_foot_ang_vel = d.cvel[left_foot_body_id][:3]
    left_foot_force = d._impl.cfrc_ext[left_foot_body_id][3:]
    left_foot_ground_contact = sim.get_collision(d, left_foot_geom_id, ground_geom_id)

    current_torque = d.ctrl[:]
    body_com = d.subtree_com[body_id]

    # ---------------- Sensor noise ----------------
    key, subkey = jax.random.split(key)
    joint_pos  = joint_pos  + sim.cfg["STD"]["std_joint_pos"] * jax.random.normal(subkey, joint_pos.shape)
    key, subkey = jax.random.split(key)
    joint_vel  = joint_vel  + sim.cfg["STD"]["std_joint_vel"] * jax.random.normal(subkey, joint_vel.shape)
    key, subkey = jax.random.split(key)
    base_ang_vel = base_ang_vel + sim.cfg["STD"]["std_gyro"] * jax.random.normal(subkey, base_ang_vel.shape)
    key, subkey = jax.random.split(key)
    base_lin_accel = base_lin_accel + sim.cfg["STD"]["std_acc"] * jax.random.normal(subkey, base_lin_accel.shape)

    # ---------------- Frames & helpers ----------------
    gravity_direction = jnp.array([0.0, 0.0, -1.0])
    body_roll, body_pitch, body_yaw = _quat_to_small_euler(body_q)
    projected_gravity = _rotate_vector_inverse_rpy(body_roll, body_pitch, body_yaw, gravity_direction)

    # rotate linear velocity into BODY frame (Isaac tracking is in base frame)
    body_vel = _rotate_vector_inverse_rpy(body_roll, body_pitch, body_yaw, body_vel_w)
    vx, vy, vz = body_vel
    base_ang_vel_b = _rotate_vector_inverse_rpy(body_roll, body_pitch, body_yaw, base_ang_vel)
    wz = base_ang_vel_b[2]
    tilt = jnp.sqrt(body_roll * body_roll + body_pitch * body_pitch)

    # contacts
    l_contact = jnp.asarray(left_foot_ground_contact,  dtype=jnp.float32)
    r_contact = jnp.asarray(right_foot_ground_contact, dtype=jnp.float32)
    flight         = (1.0 - l_contact) * (1.0 - r_contact)
    single_support = l_contact * (1.0 - r_contact) + r_contact * (1.0 - l_contact)
    double_support = l_contact * r_contact

    # gait phase (kept for obs compatibility)
    time_in_seconds = step_num * (1.0 / sim.cfg["PPO"]["model_freq"])
    gait_freq = 1.5
    gait_phase = (2 * jnp.pi * gait_freq * time_in_seconds) % (2 * jnp.pi)
    obs_gait = jnp.array([jnp.cos(gait_phase), jnp.sin(gait_phase)])

    # ---------------- Isaac-like rewards & costs ----------------
    # exp kernels
    def _expq2(x_sq, s):
        # exp( - x^2 / (2 s^2) )
        return jnp.exp(-x_sq / (2.0 * s * s + 1e-9))

    # commanded velocities (BODY frame), clipped
    cmd_xy = goal_velocity[:2]
    cmd_wz = goal_velocity[2]

    cmd_lin_mag = jnp.linalg.norm(cmd_xy)
    standing = cmd_lin_mag < 0.15
    c_move_when_standing = jnp.where(standing, jnp.sum(v_xy * v_xy), 0.0)

    cmd_xy = jnp.where(standing, cmd_xy*0, cmd_xy)

    eps = 1e-6
    v_norm   = jnp.linalg.norm(v_xy) + eps
    cmd_norm = jnp.linalg.norm(cmd_xy) + eps

    # direction-only (in [0,1])
    cos_theta = jnp.clip(jnp.dot(v_xy, cmd_xy) / (v_norm * cmd_norm), -1.0, 1.0)
    r_dir = 0.5 * (1.0 + cos_theta)

    # speed-only
    s_speed = 0.6  # will schedule later
    r_spd = jnp.exp(-((v_norm - cmd_norm) ** 2) / (2.0 * s_speed * s_speed + 1e-9))

    # blended linear tracking
    alpha_dir = 0.7  # will schedule later
    r_trk_lin_xy = alpha_dir * r_dir + (1.0 - alpha_dir) * r_spd

    # yaw-rate tracking (exp kernel)
    r_trk_ang_z = _expq2((wz - cmd_wz) ** 2, s=0.5)

    # "alive" (small positive bias for staying up)
    r_alive = (body_pos[2] > 0.20).astype(jnp.float32)

    # costs (L2)
    c_lin_vel_z = jnp.where(standing, vz*vz, 0.0)
    base_ang_vel_b = _rotate_vector_inverse_rpy(body_roll, body_pitch, body_yaw, base_ang_vel)
    c_ang_vel_xy   = base_ang_vel_b[0]**2 + base_ang_vel_b[1]**2

    # flat orientation: projected gravity XY L2 ~ 0 when upright
    c_flat_orient = projected_gravity[0] * projected_gravity[0] + projected_gravity[1] * projected_gravity[1]

    # base height L2 around target
    z_min = 0.30
    c_base_height = jnp.maximum(z_min - body_pos[2], 0.0)**2

    # torque regularizer (normalize by limit)
    tau_lim_cfg = jnp.array(sim.cfg["PPO"]["torque_limit"])
    # support lists like [-4.9, 4.9] or per-joint ranges; take max magnitude
    tau_max = jnp.max(jnp.abs(tau_lim_cfg))
    tau_max = jnp.maximum(tau_max, 1e-6)
    c_tau = jnp.mean((current_torque / tau_max) ** 2)

    # additional regularizers
    c_qd   = jnp.mean(joint_vel ** 2)
    c_act  = jnp.mean(current_action ** 2)
    c_dact = jnp.mean((current_action - prev_action) ** 2)

    # joint limits (existing)
    max_q = jnp.array(sim.cfg["PPO"]["joint_q_max"])
    min_q = jnp.array(sim.cfg["PPO"]["joint_q_min"])
    over  = jnp.clip(joint_pos - max_q, a_min=0.0)
    under = jnp.clip(min_q - joint_pos, a_min=0.0)
    joint_pos_limit = jnp.sum(over*over + under*under)

    # deviation from default pose (existing)
    c_joint_devation = jnp.linalg.norm(joint_pos - jnp.array(sim.cfg["PPO"]["default_qpos"]))

    # contact force penalty (approx of undesired contacts): penalize large impacts
    fL = jnp.linalg.norm(left_foot_force)
    fR = jnp.linalg.norm(right_foot_force)
    F_MAX = 1.5 * body_mass * 9.81  # ~1.5 x body weight
    c_contact_force = jnp.maximum(0.0, fL - F_MAX) + jnp.maximum(0.0, fR - F_MAX)

    # ---------------- Weights (per-second) -> scale by dt_model ----------------
    dt_model = 1.0 / sim.cfg["PPO"]["model_freq"]   # your policy/reward rate = 50 Hz => 0.02 s

    # ---------- POSITIVE (per-second) ----------
    w_trk_lin_ps      = 8.0      # main driver; ~0.16 per-step at dt=0.02
    w_trk_ang_ps      = 1.0      # stable turning incentive
    w_alive_ps        = 0.5      # small bias to stay upright
    w_single_ps       = 1.2      # promotes stepping when moving
    w_dsup_ps         = 0.2      # encourages double support when standing

    # ---------- NEGATIVE (per-second) ----------
    w_lin_z_ps        = 0.7      # penalize vertical motion (only when standing)
    w_ang_xy_ps       = 0.2      # resist roll/pitch angular velocity
    w_flat_ps         = 0.8      # upright posture cost
    w_hgt_ps          = 0.9      # keep COM above minimum height
    w_tau_ps          = 0.3      # effort regularizer (normalized torque)
    w_qd_ps           = 0.05     # joint velocity smoothing
    w_act_ps          = 0.002    # tiny redundancy term; may set to 0.0
    w_dact_ps         = 0.5      # action-change penalty for smoothness
    w_jlim_ps         = 2.0      # soft squared hinge joint-limit penalty
    w_jointdev_ps     = 0.05     # mild pull toward nominal pose
    w_cfor_ps         = 5e-3     # contact-force spike penalty
    w_flight_ps       = 0.3      # discourages excessive flight
    w_move_stand_ps   = 1.5      # penalize drifting when commanded to stand

    # scale by dt_model
    w_trk_lin  = w_trk_lin_ps  * dt_model
    w_trk_ang  = w_trk_ang_ps  * dt_model
    w_alive    = w_alive_ps    * dt_model
    w_single   = w_single_ps   * dt_model

    w_lin_z    = w_lin_z_ps    * dt_model
    w_ang_xy   = w_ang_xy_ps   * dt_model
    w_flat     = w_flat_ps     * dt_model
    w_hgt      = w_hgt_ps      * dt_model
    w_tau      = w_tau_ps      * dt_model
    w_qd       = w_qd_ps       * dt_model
    w_act      = w_act_ps      * dt_model
    w_dact     = w_dact_ps     * dt_model
    w_jlim     = w_jlim_ps     * dt_model
    w_jointdev = w_jointdev_ps * dt_model
    w_cfor     = w_cfor_ps     * dt_model
    w_flight   = w_flight_ps   * dt_model
    w_dsup     = w_dsup_ps     * dt_model
    w_move_stand = w_move_stand_ps * dt_model

    # effective (piecewise) support/flight weights
    w_single_eff = jnp.where(standing, -2 * w_single,  w_single)
    w_dsup_eff   = jnp.where(standing,  2 * w_dsup,   -w_dsup)

    w_flat_eff = jnp.where(standing, w_flat, 0.5 * w_flat)

    # ---------------- Assemble reward (no clamping; negatives allowed) ----------------
    # support / flight terms depend on standing vs moving
    support_reward = (
        w_single_eff  * single_support
        + w_dsup_eff    * double_support
    )

    reward_pos = (
        w_trk_lin * r_trk_lin_xy
        + w_trk_ang * r_trk_ang_z
        + w_alive   * r_alive
        + support_reward
    )

    reward_neg = (
        w_lin_z   * c_lin_vel_z
        + w_ang_xy  * c_ang_vel_xy
        + w_flat_eff    * c_flat_orient
        + w_hgt     * c_base_height
        + w_tau     * c_tau
        + w_qd      * c_qd
        + w_act     * c_act
        + w_dact    * c_dact
        + w_jlim    * joint_pos_limit
        + w_jointdev* c_joint_devation
        + w_cfor    * c_contact_force
        + w_flight  * flight
        + w_move_stand * c_move_when_standing
    )

    reward = reward_pos - reward_neg

    # ---------------- Done flags ----------------
    fallen = (body_pos[2] < 0.20)
    done = (fallen) | (step_num > sim.cfg["PPO"]["max_timesteps"])
    done = jnp.where((done == 0) & ((step_num + 1) % 100 == 0), -1, done)

    # ---------------- Obs vector ----------------
    obs = jnp.concatenate([
        goal_velocity,                 # (3,)
        projected_gravity,             # (3,)
        base_ang_vel,                  # (3,)
        joint_pos,                     # (..)
        joint_vel,                     # (..)
        prev_action,                   # (..)
        jnp.array((body_mass,)),       # (1,)
        body_com,                      # (3,)
        body_vel,                      # (3,) body-frame linear vel
        jnp.array((body_pos[2],)),     # (1,)
        push_force                     # (...)
    ], axis=-1)

    # ---------------- Logging terms ----------------
    reward_terms = FrozenDict({
        "full_reward":       reward,

        # positive
        "r_trk_lin_xy":      w_trk_lin * r_trk_lin_xy,
        "r_trk_ang_z":       w_trk_ang * r_trk_ang_z,
        "r_alive":           w_alive   * r_alive,

        # support
        "r_single":          w_single_eff * single_support,
        "r_double":          w_dsup_eff   * double_support,

        # negative
        "c_lin_vel_z":       -w_lin_z   * c_lin_vel_z,
        "c_ang_vel_xy":      -w_ang_xy  * c_ang_vel_xy,
        "c_flat_orient":     -w_flat_eff    * c_flat_orient,
        "c_base_height":     -w_hgt     * c_base_height,
        "c_tau":             -w_tau     * c_tau,
        "c_qd":              -w_qd      * c_qd,
        "c_act":             -w_act     * c_act,
        "c_dact":            -w_dact    * c_dact,
        "joint_pos_limit":   -w_jlim    * joint_pos_limit,
        "c_joint_dev":       -w_jointdev* c_joint_devation,
        "c_contact_force":   -w_cfor    * c_contact_force,
        "flight":            -w_flight * flight,
        # optional helpers
        "cmd_vx": cmd_xy[0], "cmd_vy": cmd_xy[1], "cmd_wz": cmd_wz,
        "vx": vx, "vy": vy, "vz": vz, "wz": wz,
    })

    return obs, reward, done, reward_terms

