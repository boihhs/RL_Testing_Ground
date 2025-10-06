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
    # ---------------- Helpers ----------------
    def _quat_to_small_euler(q):
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        yaw   = jnp.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        pitch = jnp.arcsin(jnp.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
        roll  = jnp.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        return roll, pitch, yaw

    def _rotate_vector_inverse_rpy(roll, pitch, yaw, v):
        R_x = jnp.array([[1, 0, 0],
                         [0, jnp.cos(roll), -jnp.sin(roll)],
                         [0, jnp.sin(roll),  jnp.cos(roll)]])
        R_y = jnp.array([[ jnp.cos(pitch), 0, jnp.sin(pitch)],
                         [0,               1,               0],
                         [-jnp.sin(pitch), 0, jnp.cos(pitch)]])
        R_z = jnp.array([[ jnp.cos(yaw), -jnp.sin(yaw), 0],
                         [ jnp.sin(yaw),  jnp.cos(yaw), 0],
                         [0, 0, 1]])
        return (R_z @ R_y @ R_x).T @ v  # world->body

    def _quat_rotate(q, v):
        # Rotate vector v by quaternion q = [w,x,y,z]
        w, x, y, z = q
        q_xyz = jnp.array([x, y, z])
        t = 2.0 * jnp.cross(q_xyz, v)
        return v + w * t + jnp.cross(q_xyz, t)

    def _expq2(x_sq, s):
        return jnp.exp(-x_sq / (2.0 * s * s + 1e-9))

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
    right_foot_q   = d.xquat[right_foot_body_id]
    right_foot_vel_w = d.cvel[right_foot_body_id][3:]

    left_foot_pos = d.xpos[left_foot_body_id]
    left_foot_q   = d.xquat[left_foot_body_id]
    left_foot_vel_w  = d.cvel[left_foot_body_id][3:]

    right_foot_force = d._impl.cfrc_ext[right_foot_body_id][3:]
    left_foot_force = d._impl.cfrc_ext[left_foot_body_id][3:]

    right_foot_ground_contact = sim.get_collision(d, right_foot_geom_id, ground_geom_id)
    left_foot_ground_contact  = sim.get_collision(d, left_foot_geom_id, ground_geom_id)

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

    # rotate linear velocity into BODY frame (full RPY)
    body_vel = _rotate_vector_inverse_rpy(body_roll, body_pitch, body_yaw, body_vel_w)
    vx, vy, vz = body_vel

    base_ang_vel_b = _rotate_vector_inverse_rpy(body_roll, body_pitch, body_yaw, base_ang_vel)
    wz_world = base_ang_vel[2]

    # contacts
    l_contact = jnp.asarray(left_foot_ground_contact,  dtype=jnp.float32)
    r_contact = jnp.asarray(right_foot_ground_contact, dtype=jnp.float32)
    flight         = (1.0 - l_contact) * (1.0 - r_contact)
    single_support = l_contact * (1.0 - r_contact) + r_contact * (1.0 - l_contact)
    double_support = l_contact * r_contact

    # ---------------- Commands & tracking ----------------
    cmd_xy = goal_velocity[:2]       # body-frame desired XY
    cmd_wz = goal_velocity[2]        # desired yaw rate (we'll compare to world-z)

    vel_rewd = 3 * _expq2(jnp.linalg.norm(cmd_xy - body_vel[:2]), .2)
    defalt_pos_reward = _expq2(jnp.linalg.norm(joint_pos - jnp.array(sim.cfg["PPO"]["default_qpos"])), .6)

    torque_reward = _expq2(jnp.linalg.norm(current_torque), .6)

    reward = single_support + vel_rewd + defalt_pos_reward + torque_reward

    # ---------------- Done flags ----------------
    fallen = (body_pos[2] < 0.20)
    done = (fallen) | (step_num > sim.cfg["PPO"]["max_timesteps"])
    done = jnp.where((done == 0) & ((step_num + 1) % 200 == 0), -1, done)

    # ---------------- Obs vector ----------------
    obs = jnp.concatenate([
        goal_velocity,                 # (3,)
        projected_gravity,             # (3,)
        base_ang_vel,                  # (3,) world ang vel (kept)
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
        "vel reward" :       vel_rewd,
        "defalt_pos_reward": defalt_pos_reward,
        "torque_reward": torque_reward

        
    })

    return obs, reward, done, reward_terms