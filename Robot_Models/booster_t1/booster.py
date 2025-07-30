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
import numpy as np
from mujoco import mjx
import mujoco

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Sensor:
    id: str     = field(metadata={'static': True})
    start: int     = field(metadata={'static': True})
    length: int     = field(metadata={'static': True})


@jax.jit
def get_obs_and_reward_walking(env, sim, key):

    def _quat_to_small_euler(q):
        qw, qx, qy, qz = q
        # Yaw (around z-axis)
        yaw = jnp.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        # Pitch (around y-axis)
        pitch = jnp.arcsin(jnp.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
        # Roll (around x-axis)
        roll = jnp.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        return pitch, roll, yaw
    
    # Get obs things
    d = env.mjx_data
    step_num = env.step_num
    current_action = env.curr_action
    prev_action = env.prev_action
    goal_velocity = env.goal_velocity

    ang_vel_id = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
    ang_vel_sensor = Sensor(ang_vel_id, sim.mjx_model.sensor_adr[ang_vel_id], sim.mjx_model.sensor_dim[ang_vel_id])

    lin_accel_id = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_SENSOR, "linear-acceleration")
    lin_accel_sensor = Sensor(lin_accel_id, sim.mjx_model.sensor_adr[lin_accel_id], sim.mjx_model.sensor_dim[lin_accel_id])

    right_foot_body_id = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")
    left_foot_body_id = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
    body_id = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")

    right_foot_geom_id = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom")
    left_foot_geom_id = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom")
    ground_geom_id = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    body_pos = d.qpos[:3]
    quat = d.qpos[3:7]
    body_vel = d.qvel[:3]
    joint_pos = d.qpos[7:]
    joint_vel = d.qvel[6:]

    base_ang_vel = d.sensordata[ang_vel_sensor.start:ang_vel_sensor.start + ang_vel_sensor.length]
    base_lin_accel = d.sensordata[lin_accel_sensor.start:lin_accel_sensor.start + lin_accel_sensor.length]

    right_foot_pos = d.xpos[right_foot_body_id]
    right_foot_vel = d.cvel[right_foot_body_id][3:]
    right_foot_ang = d.cvel[right_foot_body_id][:3]
    right_foot_ang_vel = d.cvel[right_foot_body_id][:3]
    right_foot_force = d._impl.cfrc_ext[right_foot_body_id][3:]
    right_foot_moment = d._impl.cfrc_ext[right_foot_body_id][:3]
    right_foot_ground_contact = sim.get_collision(d, right_foot_geom_id, ground_geom_id)

    left_foot_pos = d.xpos[left_foot_body_id]
    left_foot_vel = d.cvel[left_foot_body_id][3:]
    left_foot_ang = d.cvel[left_foot_body_id][:3]
    left_foot_ang_vel = d.cvel[left_foot_body_id][:3]
    left_foot_force = d._impl.cfrc_ext[left_foot_body_id][3:]
    left_foot_ang_moment = d._impl.cfrc_ext[left_foot_body_id][:3]
    left_foot_ground_contact = sim.get_collision(d, left_foot_geom_id, ground_geom_id)

    prev_ctrl = d.ctrl[:]
    robot_com = d.subtree_com[body_id]

    key, subkey = jax.random.split(key)
    noise_joint_pos = sim.cfg["STD"]["std_joint_pos"] * jax.random.normal(subkey, joint_pos.shape)
    joint_pos = joint_pos + noise_joint_pos

    key, subkey = jax.random.split(key)
    noise_joint_vel = sim.cfg["STD"]["std_joint_vel"] * jax.random.normal(subkey, joint_vel.shape)
    joint_vel = joint_vel + noise_joint_vel

    key, subkey = jax.random.split(key)
    noise_ang_vel = sim.cfg["STD"]["std_gyro"] * jax.random.normal(subkey, base_ang_vel.shape)
    base_ang_vel = base_ang_vel + noise_ang_vel

    key, subkey = jax.random.split(key)
    noise_lin_accel = sim.cfg["STD"]["std_acc"] * jax.random.normal(subkey, base_lin_accel.shape)
    base_lin_accel = base_lin_accel + noise_lin_accel

    gait_freq = 2

    time_in_seconds = step_num * (1 / sim.cfg["PPO"]["model_freq"])
    gait_phase = 2 * jnp.pi * gait_freq * time_in_seconds
    obs_gait = jnp.array([
        jnp.cos(gait_phase),
        jnp.sin(gait_phase)
    ])

    # Reward things
    goal_pos = jnp.array([0, 1, .63])

    joint_pos_upper = jnp.array(sim.cfg["PPO"]["joint_q_max"])
    joint_pos_lower = jnp.array(sim.cfg["PPO"]["joint_q_min"])
    tau_limit = jnp.array(sim.cfg["PPO"]["torque_limit"])
    
    SIGMA_X   = 0.25
    SIGMA_Y   = 0.25
    DES_HEIGHT = 0.63        # torso height set‑point         [m]
    D_REF      = 0.22        # desired foot spacing           [m]
    H_LIFT     = 0.05        # ≥ this ⇒ foot counted “swing”  [m]
   
    v_des_xy = jnp.array([0.0, 0.10])          # 0.10 m/s forward
    v_xy     = body_vel[:2]
    rew_track = 3.0 * jnp.exp(-jnp.linalg.norm(v_xy - v_des_xy)**2 / 0.02)
    
    # survival bonus + torso–height term
    rew_track += 0.025
    rew_track += -10.0 * (DES_HEIGHT - robot_com[2])**2 

    pitch, roll, yaw_base = _quat_to_small_euler(quat)
    rew_track += -8.0 * (pitch**2 + roll**2)

    left_off  = 1 - right_foot_ground_contact
    right_off = 1 - left_foot_ground_contact

    stance_mask   = jnp.stack([1.0 - left_off, 1.0 - right_off])          # 1 if foot on ground
    stance_feet   = jnp.stack([left_foot_pos[:2], right_foot_pos[:2]])
    support_cent  = jnp.sum(stance_feet * stance_mask[:, None], axis=0) \
                    / (jnp.sum(stance_mask) + 1e-6)
    rew_track    += -5.0 * jnp.linalg.norm(robot_com[:2] - support_cent)    # COM alignment

    phase              = (2 * jnp.pi * gait_freq * time_in_seconds) % (2 * jnp.pi)
    left_should_swing  = phase < jnp.pi
    right_should_swing = ~left_should_swing

    rew_gait  = 3.0 * (
        left_should_swing.astype(jnp.float32)  * left_off.astype(jnp.float32)
        + right_should_swing.astype(jnp.float32) * right_off.astype(jnp.float32)
    )

    foot_clear  = (
        left_should_swing.astype(jnp.float32)  * (left_foot_pos[2]  > H_LIFT)
      + right_should_swing.astype(jnp.float32) * (right_foot_pos[2] > H_LIFT)
    )
    rew_gait += 2.0 * foot_clear

    rew_gait += -0.1 * (
        (1.0 - left_off.astype(jnp.float32))  * jnp.sum(left_foot_vel[:2]**2)
        + (1.0 - right_off.astype(jnp.float32)) * jnp.sum(right_foot_vel[:2]**2)
    )

    d_feet  = jnp.linalg.norm(right_foot_pos[:2] - left_foot_pos[:2])
    rew_gait += -1.0 * jnp.maximum(D_REF - d_feet, 0.0)

    tau      = prev_ctrl           # your applied torques
    tau_max  = tau_limit
    rew_reg  = (
        -2e-4 * jnp.sum(tau**2)
    - 1e-2 * jnp.sum((tau / tau_max)**2)
    - 5e-4 * jnp.sum(jnp.maximum(tau * joint_vel, 0.0))   # positive mechanical power
    - 2.0  * body_vel[2]**2
    - 0.3  * jnp.sum(base_ang_vel[:2]**2)          # slightly stronger
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
    | (jnp.abs(roll)  > jnp.deg2rad(20)) | (step_num > sim.cfg["PPO"]["max_timesteps"])
    )
    support_dist = jnp.linalg.norm(robot_com[:2] - support_cent)
    fail = fail | (support_dist > 0.2)
    done = fail.astype(jnp.float32)

    obs = jnp.concatenate([
        prev_ctrl, joint_pos, joint_vel, base_ang_vel, base_lin_accel, quat, obs_gait,
        body_pos, body_vel,
        right_foot_pos, left_foot_pos,
        right_foot_vel, left_foot_vel,
        right_foot_ang_vel, left_foot_ang_vel,
        right_foot_force, left_foot_force,
        right_foot_moment, left_foot_ang_moment,
        jnp.array([right_foot_ground_contact, left_foot_ground_contact]), robot_com
    ], axis=-1)
    
    return obs, reward, done


        