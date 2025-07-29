import jax.numpy as jnp, jax
from jax import random
from jax import lax
from jax.tree_util import register_pytree_node_class
from functools import partial
from Mujoco_Env.Sim import Sim, SimCfg, ENVS
import optax
from flax.training import checkpoints
from pathlib import Path
from dataclasses import dataclass, field
from Models.Policy import Policy
from Models.Value import Value
from Buffer.Buffer import ReplayBuffer
import yaml
import matplotlib.pyplot as plt
from jax import debug
import numpy as np


def get_obs(d, step_num, sim: Sim):

    body_pos = d.qpos[:3]
    quat = d.qpos[3:7]
    body_vel = d.qvel[:3]
    joint_pos = d.qpos[7:]
    joint_vel = d.qvel[6:]

    base_ang_vel = d.sensordata[sim.ang_vel_sensor.start:sim.ang_vel_sensor.start + sim.ang_vel_sensor.length]
    base_lin_accel = d.sensordata[sim.lin_accel_sensor.start:sim.lin_accel_sensor.start + sim.lin_accel_sensor.length]

    right_foot_pos = d.xpos[sim.right_foot_body_id]
    right_foot_vel = d.cvel[sim.right_foot_body_id][3:]
    right_foot_ang_vel = d.cvel[sim.right_foot_body_id][:3]
    right_foot_force = d._impl.cfrc_ext[sim.right_foot_body_id][3:]
    right_foot_moment = d._impl.cfrc_ext[sim.right_foot_body_id][:3]
    right_foot_ground_contact = sim.get_collision(d, sim.right_foot_geom_id, sim.ground_geom_id)

    left_foot_pos = d.xpos[sim.left_foot_body_id]
    left_foot_vel = d.cvel[sim.left_foot_body_id][3:]
    left_foot_ang_vel = d.cvel[sim.left_foot_body_id][:3]
    left_foot_force = d._impl.cfrc_ext[sim.left_foot_body_id][3:]
    left_foot_ang_moment = d._impl.cfrc_ext[sim.left_foot_body_id][:3]
    left_foot_ground_contact = sim.get_collision(d, sim.left_foot_geom_id, sim.ground_geom_id)

    prev_ctrl = d.ctrl[:]

    key, subkey = jax.random.split(key)
    noise_joint_pos = sim.cfg.std_joint_pos * jax.random.normal(subkey, joint_pos.shape)
    joint_pos = joint_pos + noise_joint_pos

    key, subkey = jax.random.split(key)
    noise_joint_vel = sim.cfg.std_joint_vel * jax.random.normal(subkey, joint_vel.shape)
    joint_vel = joint_vel + noise_joint_vel

    key, subkey = jax.random.split(key)
    noise_ang_vel = sim.cfg.std_gyro * jax.random.normal(subkey, base_ang_vel.shape)
    base_ang_vel = base_ang_vel + noise_ang_vel

    key, subkey = jax.random.split(key)
    noise_lin_accel = sim.cfg.std_acc * jax.random.normal(subkey, base_lin_accel.shape)
    base_lin_accel = base_lin_accel + noise_lin_accel

    gait_frequency = 2

    time_in_seconds = step_num * (1 / sim.cfg.model_freq)
    gait_phase = 2 * jnp.pi * gait_frequency * time_in_seconds
    obs_gait = jnp.array([
        jnp.cos(gait_phase),
        jnp.sin(gait_phase)
    ])

    obs = jnp.concatenate([
        prev_ctrl, joint_pos, joint_vel, base_ang_vel, base_lin_accel, quat, obs_gait,
        body_pos, body_vel,
        right_foot_pos, left_foot_pos,
        right_foot_vel, left_foot_vel,
        right_foot_ang_vel, left_foot_ang_vel,
        right_foot_force, left_foot_force,
        right_foot_moment, left_foot_ang_moment,
        jnp.array([right_foot_ground_contact, left_foot_ground_contact])
    ], axis=-1)
    
    return obs


def reward_walking(obs, actions, step_counter, prev_action, cfg):
    def _quat_to_small_euler(q):
        qw, qx, qy, qz = q
        # Yaw (around z-axis)
        yaw = jnp.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        # Pitch (around y-axis)
        pitch = jnp.arcsin(jnp.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
        # Roll (around x-axis)
        roll = jnp.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        return pitch, roll, yaw
    
    current_action = actions
    prev_action = prev_action
    v_cmd = jnp.array([0, .1, 0])
    time = step_counter * (1 / cfg["PPO"]["model_freq"])
    current_ctrl           = obs[:23]
    joint_pos              = obs[23:46]
    joint_vel              = obs[46:69]
    base_ang_vel           = obs[69:72]
    base_lin_accel         = obs[72:75]
    quat                   = obs[75:79]
    obs_gait               = obs[79:81]
    body_pos               = obs[81:84]
    body_vel               = obs[84:87]
    right_foot_pos         = obs[87:90]
    left_foot_pos          = obs[90:93]
    right_foot_vel         = obs[93:96]
    left_foot_vel          = obs[96:99]
    right_foot_ang_vel     = obs[99:102]
    left_foot_ang_vel      = obs[102:105]
    right_foot_force       = obs[105:108]
    left_foot_force        = obs[108:111]
    right_foot_moment      = obs[111:114]
    left_foot_ang_moment   = obs[114:117]
    right_foot_ground_contact = obs[117]
    left_foot_ground_contact  = obs[118]

    joint_pos_upper = jnp.array(cfg["PPO"]["joint_q_max"])
    joint_pos_lower = jnp.array(cfg["PPO"]["joint_q_min"])
    tau_limit = jnp.array(cfg["PPO"]["torque_limit"])
    gait_freq = 2

    SIGMA_X   = 0.25
    SIGMA_Y   = 0.25
    DES_HEIGHT = 0.65        # torso height set‑point         [m]
    D_REF      = 0.22        # desired foot spacing           [m]
    H_LIFT     = 0.05        # ≥ this ⇒ foot counted “swing”  [m]
   
    prog_err = body_pos[1] - v_cmd[1] * time          # torso should have advanced
    rew_track  = (
        jnp.exp(-((v_cmd[0] - body_vel[0])**2) / SIGMA_X)
        + jnp.exp(-((v_cmd[1] - body_vel[1])**2) / SIGMA_Y)
        + 0.5 * jnp.exp(-((v_cmd[2] - base_ang_vel[2])**2) / SIGMA_X)
    ) - 10.0 * prog_err**2 
    # survival bonus + torso–height term
    rew_track += 0.025
    rew_track += -20.0 * (DES_HEIGHT - body_pos[2])**2

    pitch, roll, yaw_base = _quat_to_small_euler(quat)
    rew_track += -5.0 * (pitch**2 + roll**2)

    left_off  = 1 - right_foot_ground_contact
    right_off = 1 - left_foot_ground_contact

    stance_mask   = jnp.stack([1.0 - left_off, 1.0 - right_off])          # 1 if foot on ground
    stance_feet   = jnp.stack([left_foot_pos[:2], right_foot_pos[:2]])
    support_cent  = jnp.sum(stance_feet * stance_mask[:, None], axis=0) \
                    / (jnp.sum(stance_mask) + 1e-6)
    rew_track    += -5.0 * jnp.linalg.norm(body_pos[:2] - support_cent)    # COM alignment

    phase              = (2 * jnp.pi * gait_freq * time) % (2 * jnp.pi)
    left_should_swing  = phase < jnp.pi
    right_should_swing = ~left_should_swing

    rew_gait  = 3.0 * (
        left_should_swing.astype(jnp.float32)  * left_off.astype(jnp.float32)
        + right_should_swing.astype(jnp.float32) * right_off.astype(jnp.float32)
    )

    rew_gait += -0.1 * (
        (1.0 - left_off.astype(jnp.float32))  * jnp.sum(left_foot_vel[:2]**2)
        + (1.0 - right_off.astype(jnp.float32)) * jnp.sum(right_foot_vel[:2]**2)
    )

    d_feet  = jnp.linalg.norm(right_foot_pos[:2] - left_foot_pos[:2])
    rew_gait += -1.0 * jnp.maximum(D_REF - d_feet, 0.0)

    tau      = current_ctrl           # your applied torques
    tau_max  = tau_limit
    rew_reg  = (
        -2e-4 * jnp.sum(tau**2)
    - 1e-2 * jnp.sum((tau / tau_max)**2)
    - 2e-4 * jnp.sum(jnp.maximum(tau * joint_vel, 0.0))   # positive mechanical power
    - 2.0  * body_vel[2]**2
    - 0.3  * jnp.sum(base_ang_vel[:2]**2)          # slightly stronger
    - 1.0  * body_vel[2]**2 
    - 1e-4 * jnp.sum(joint_vel**2)
    - 1e-4 * jnp.sum(base_lin_accel**2)
    - 1.0  * jnp.sum((current_action - prev_action)**2)
    )

    limit_violation = jnp.sum(
        jnp.where(joint_pos > joint_pos_upper, 1.0, 0.0)
        + jnp.where(joint_pos < joint_pos_lower, 1.0, 0.0)
    )
    rew_reg += -1.0 * limit_violation

    reward = rew_track + rew_gait + rew_reg

    fail  = (
        (body_pos[2] < 0.5)
    | (jnp.abs(pitch) > jnp.deg2rad(20))
    | (jnp.abs(roll)  > jnp.deg2rad(20)) | (step_counter > cfg["PPO"]["max_timesteps"])
    )
    done = fail.astype(jnp.float32)

    return reward, done
        