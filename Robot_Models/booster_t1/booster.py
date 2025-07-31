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
    
    # Get obs things
    d = env.mjx_data
    step_num = env.step_num
    current_action = env.curr_action
    prev_action = env.prev_action
    # (vx, vy, w_yaw)
    goal_velocity = env.goal_velocity
    push_force = env.force_applied
    body_mass = 66

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
    body_q = d.qpos[3:7]
    body_vel = d.qvel[:3]
    joint_pos = d.qpos[7:]
    joint_vel = d.qvel[6:]
    joint_accel = d.qacc[6:]

    base_ang_vel = d.sensordata[ang_vel_sensor.start:ang_vel_sensor.start + ang_vel_sensor.length]
    base_lin_accel = d.sensordata[lin_accel_sensor.start:lin_accel_sensor.start + lin_accel_sensor.length]
    base_ang_accel = d.qacc[3:6]

    right_foot_pos = d.xpos[right_foot_body_id]
    right_foot_vel = d.cvel[right_foot_body_id][3:]
    right_foot_q = d.xquat[right_foot_body_id]
    right_foot_ang_vel = d.cvel[right_foot_body_id][:3]
    right_foot_force = d._impl.cfrc_ext[right_foot_body_id][3:]
    right_foot_moment = d._impl.cfrc_ext[right_foot_body_id][:3]
    right_foot_ground_contact = sim.get_collision(d, right_foot_geom_id, ground_geom_id)

    left_foot_pos = d.xpos[left_foot_body_id]
    left_foot_vel = d.cvel[left_foot_body_id][3:]
    left_foot_q = d.xquat[left_foot_body_id]
    left_foot_ang_vel = d.cvel[left_foot_body_id][:3]
    left_foot_force = d._impl.cfrc_ext[left_foot_body_id][3:]
    left_foot_ang_moment = d._impl.cfrc_ext[left_foot_body_id][:3]
    left_foot_ground_contact = sim.get_collision(d, left_foot_geom_id, ground_geom_id)

    current_torque = d.ctrl[:]
    body_com = d.subtree_com[body_id]

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

    gravity_direction = jnp.array([.0, .0, -1.])

    body_roll, body_pitch, body_yaw = _quat_to_small_euler(body_q)
    left_foot_roll, left_foot_pitch, left_foot_yaw = _quat_to_small_euler(left_foot_q)
    right_foot_roll, right_foot_pitch, right_foot_yaw = _quat_to_small_euler(right_foot_q)

    projected_gravity = _rotate_vector_inverse_rpy(body_roll, body_pitch, body_yaw, gravity_direction)

    gait_freq = 1.5
    h_des = 0.68
    tau_limit = jnp.array(sim.cfg["PPO"]["torque_limit"])
    max_q = jnp.array(sim.cfg["PPO"]["joint_q_max"])
    min_q = jnp.array(sim.cfg["PPO"]["joint_q_min"])


    time_in_seconds = step_num * (1 / sim.cfg["PPO"]["model_freq"])
    gait_phase = (2 * jnp.pi * gait_freq * time_in_seconds) % (2 * jnp.pi)
    obs_gait = jnp.array([
        jnp.cos(gait_phase),
        jnp.sin(gait_phase)
    ])

    # Reward things (no collison)

    left_should_swing  = (gait_phase < jnp.pi).astype(jnp.float32)
    right_should_swing = 1.0 - left_should_swing

    survival = 0.025
    vel_track_x = jnp.exp(-(goal_velocity[0] - body_vel[0])**2/ .25)
    vel_track_y = jnp.exp(-(goal_velocity[1] - body_vel[1])**2/ .25)
    vel_track_yaw = jnp.exp(-(goal_velocity[2] - base_ang_vel[2])**2/ .25) * .5
    base_height = (h_des - body_pos[2])**2 * -20
    orientation = (body_roll**2 + body_pitch**2) * -5
    torque = jnp.linalg.norm(current_torque)**2 * -2e-4
    torque_tiredness = jnp.linalg.norm(current_torque / tau_limit)**2 * -1e-2
    power = jnp.maximum(jnp.sum(current_torque * joint_vel), 0) * -2e-4
    lin_vel = body_vel[2]**2 * -2.
    ang_vel_xy = jnp.linalg.norm(base_ang_vel[:2])**2 * -.2
    joint_vel_reward = jnp.linalg.norm(joint_vel)**2 * -1e-4
    joint_accel_reward = jnp.linalg.norm(joint_accel)**2 * -1e-7
    base_accel = (jnp.linalg.norm(base_lin_accel)**2 + jnp.linalg.norm(base_ang_accel)**2) * -1e-4
    action_rate = jnp.linalg.norm(current_action - prev_action)**2 * -1
    joint_pos_limit = jnp.sum(jnp.where(joint_pos > max_q, 1, 0) + jnp.where(joint_pos < min_q, 1, 0)) * -1
    feet_swing = (left_should_swing * (1 - left_foot_ground_contact) + right_should_swing * (1 - right_foot_ground_contact)) * 3
    feet_slip = ((1 - left_should_swing) * jnp.linalg.norm(left_foot_vel[:2])**2 + (1 - right_should_swing) * jnp.linalg.norm(right_foot_vel[:2])**2) * -.1
    feet_yaw = ((left_foot_yaw - body_yaw)**2 + (right_foot_yaw - body_yaw)**2) * -1
    feet_roll = (left_foot_roll**2 + right_foot_roll**2) * -.1
    d_feet   = jnp.linalg.norm(left_foot_pos[:2] - right_foot_pos[:2])
    feet_distance = -jnp.maximum(0.2 - d_feet, 0.0)

    reward = (survival + vel_track_x + vel_track_y + vel_track_yaw + base_height + orientation +
              torque + torque_tiredness + power + lin_vel + ang_vel_xy + joint_vel_reward + joint_accel_reward +
              base_accel + action_rate + joint_pos_limit + feet_swing + feet_slip + feet_yaw + feet_roll+ 
              feet_distance)
    
    
    reward = jnp.maximum(reward, 0)
            
    fallen = (body_pos[2] < .45) | (body_roll > jnp.rad2deg(20)) | (body_pitch > jnp.rad2deg(20))

    done = (fallen) | (step_num > sim.cfg["PPO"]["max_timesteps"])

    # get obs (no push torque)
    obs = jnp.concatenate([goal_velocity, obs_gait, projected_gravity, base_ang_vel, joint_pos, joint_vel, prev_action,
          jnp.array((body_mass,)), body_com, body_vel, jnp.array((body_pos[2],)), push_force], axis=-1)
    
    return obs, reward, done
