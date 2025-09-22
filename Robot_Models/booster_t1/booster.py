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
from jax import debug
from mujoco import mjx
import mujoco
from Get_IL_Data.get_ground_truth_pos import getGroundTurthPositions

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Sensor:
    id: str     = field(metadata={'static': True})
    start: int     = field(metadata={'static': True})
    length: int     = field(metadata={'static': True})

cfg_file = "/home/leo-benaharon/Desktop/HumanPose/RL_Testing_Ground/RL_Algos/PPO.yaml"
robot_data_path = "/home/leo-benaharon/Desktop/HumanPose/GMR/path_to_save_robot_data.pkl"

groundTruth_pos = getGroundTurthPositions(robot_data_path, cfg_file, .05)
groundTruth_pos.run()

@jax.jit
def get_obs_and_reward_walking(env, sim, key):

    def _quat_to_small_euler(q):
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        yaw   = jnp.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        pitch = jnp.arcsin(jnp.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
        roll  = jnp.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        return roll, pitch, yaw
    
    def _quat_to_rotmat(q):
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]

        r00 = 1 - 2*(qy*qy + qz*qz)
        r01 = 2*(qx*qy - qw*qz)
        r02 = 2*(qx*qz + qw*qy)

        r10 = 2*(qx*qy + qw*qz)
        r11 = 1 - 2*(qx*qx + qz*qz)
        r12 = 2*(qy*qz - qw*qx)

        r20 = 2*(qx*qz - qw*qy)
        r21 = 2*(qy*qz + qw*qx)
        r22 = 1 - 2*(qx*qx + qy*qy)

        return jnp.array([[r00, r01, r02],
                        [r10, r11, r12],
                        [r20, r21, r22]])
    
    def _quat_conj(q):
        qw, qx, qy, qz = q
        return jnp.array([qw, -qx, -qy, -qz])

    def _quat_mul(q1, q2):
        w1,x1,y1,z1 = q1
        w2,x2,y2,z2 = q2
        return jnp.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def _quat_log_small(q):
        # Map unit quaternion to so(3) vector (axis*angle)
        w, x, y, z = q
        v = jnp.array([x, y, z])
        nv = jnp.linalg.norm(v) + 1e-12
        # clamp w to [-1,1] to avoid NaNs
        w = jnp.clip(w, -1.0, 1.0)
        theta = 2.0 * jnp.arctan2(nv, w)
        axis = v / nv
        return theta * axis  # shape (3,)
    
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
    start_body_pos = env.start_body_pos
    start_body_ang = env.start_body_ang
    push_force = env.force_applied

    ang_vel_id = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
    ang_vel_sensor = Sensor(ang_vel_id, sim.mjx_model.sensor_adr[ang_vel_id], sim.mjx_model.sensor_dim[ang_vel_id])

    lin_accel_id = mjx.name2id(sim.mjx_model, mujoco.mjtObj.mjOBJ_SENSOR, "linear-acceleration")
    lin_accel_sensor = Sensor(lin_accel_id, sim.mjx_model.sensor_adr[lin_accel_id], sim.mjx_model.sensor_dim[lin_accel_id])

    body_pos = d.qpos[:3]
    body_q = d.qpos[3:7]
    body_vel = d.qvel[:3]
    joint_pos = d.qpos[7:]
    joint_vel = d.qvel[6:]
    joint_accel = d.qacc[6:]
    joint_body_pos = d.xpos[2:]
    joint_body_vel = d.cvel[2:, 3:]
    joint_body_ang = d.xquat[2:]
    joint_body_ang_vel = d.cvel[2:, :3]

    i = step_num % (groundTruth_pos.addon_length + groundTruth_pos.target_length)
    goal_joint_pos = groundTruth_pos.dof_pos[i]
    goal_joint_vel = groundTruth_pos.dof_vel[i]
    goal_body_pos = groundTruth_pos.body_positions[i, 0]
    goal_body_vel = groundTruth_pos.body_velcitys[i, 0]
    goal_joint_body_pos = groundTruth_pos.body_positions[i, 1:]
    goal_joint_body_vel = groundTruth_pos.body_velcitys[i, 1:]
    
    goal_body_ang = groundTruth_pos.body_rotations_quat[i, 0]
    goal_body_ang_vel = groundTruth_pos.body_ang_velitys[i, 0]
    goal_joint_body_ang = groundTruth_pos.body_rotations_quat[i, 1:]
    goal_joint_body_ang_vel = groundTruth_pos.body_ang_velitys[i, 1:]

    base_ang_vel = d.sensordata[ang_vel_sensor.start:ang_vel_sensor.start + ang_vel_sensor.length]
    base_lin_accel = d.sensordata[lin_accel_sensor.start:lin_accel_sensor.start + lin_accel_sensor.length]
    base_ang_accel = d.qacc[3:6]

    current_torque = d.ctrl[:]

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
    projected_gravity = _rotate_vector_inverse_rpy(body_roll, body_pitch, body_yaw, gravity_direction)

   
    tau_limit = jnp.array(sim.cfg["PPO"]["torque_limit"])
    max_q = jnp.array(sim.cfg["PPO"]["joint_q_max"])
    min_q = jnp.array(sim.cfg["PPO"]["joint_q_min"])


    time_in_seconds = step_num * (1 / sim.cfg["PPO"]["model_freq"])
   
    # Reward things (no collison)
    survival = 0.025
    R_b = _quat_to_rotmat(body_q)
    R_b_0 = _quat_to_rotmat(start_body_ang)
    R_b_goal = _quat_to_rotmat(goal_body_ang)
    R_b_goal_0 = _quat_to_rotmat(groundTruth_pos.body_rotations_quat[0, 0])

    # _quat_to_rotmat_batch = jax.vmap(_quat_to_rotmat)
    # R_b_bodys = _quat_to_rotmat_batch(joint_body_ang)
    # R_b_bodys_goal = _quat_to_rotmat_batch(goal_joint_body_ang)
    
    
    body_pos_error = ((goal_body_pos - groundTruth_pos.body_positions[0, 0]) @ R_b_goal_0  - (body_pos - start_body_pos) @ R_b_0) 
    body_vel_error = (goal_body_vel @ R_b_goal  - body_vel @ R_b)

    body_ang_error = (3 - jnp.trace((R_b_goal.T @ R_b_goal_0).T @ (R_b.T @ R_b_0))) / 4
    body_ang_vel_error = (goal_body_ang_vel @ R_b_goal  - base_ang_vel @ R_b)

    joint_body_pos_error = ((goal_joint_body_pos - goal_body_pos) @ R_b_goal  - (joint_body_pos - body_pos) @ R_b) 
    joint_body_vel_error = ((goal_joint_body_vel) @ R_b_goal  - (joint_body_vel) @ R_b)

    # joint_body_ang_error = (jnp.clip((jnp.trace((R_b_goal.T @ R_b_bodys_goal).T @ (R_b.T @ R_b_bodys)) - 1) * .5, -1.0, 1.0) * .5) / .3
    # joint_body_ang_vel_error = ((goal_joint_body_ang_vel) @ R_b_goal  - (joint_body_ang_vel) @ R_b)

    # errors = jnp.exp(-jnp.linalg.norm(body_pos_error) / 2) + jnp.exp(-jnp.linalg.norm(body_vel_error) / 2) + jnp.exp(-jnp.linalg.norm(joint_pos_error) / 2) + jnp.exp(-jnp.linalg.norm(joint_vel_error) / 2) + jnp.exp(-jnp.linalg.norm(joint_body_pos_error) / 2) + jnp.exp(-jnp.linalg.norm(joint_body_vel_error) / 2)
    errors = (jnp.exp(-jnp.linalg.norm(body_pos_error) / .25) + 
              jnp.exp(-jnp.linalg.norm(body_vel_error) / .25) + 
              jnp.exp(-jnp.linalg.norm(joint_body_pos_error) / .25) + 
              jnp.exp(-jnp.linalg.norm(joint_body_vel_error) / .25) + 
              jnp.exp(-jnp.linalg.norm(joint_body_vel_error) / .25) + 
              jnp.exp(-jnp.linalg.norm(body_ang_error) / .25))
            #   joint_body_ang_error + 
            #   jnp.exp(-jnp.linalg.norm(joint_body_ang_vel_error) / .25))
    
    torque = jnp.linalg.norm(current_torque)**2 * -2e-4
    torque_tiredness = jnp.linalg.norm(current_torque / tau_limit)**2 * -1e-2
    power = jnp.maximum(jnp.sum(current_torque * joint_vel), 0) * -2e-4
    
    joint_accel_reward = jnp.linalg.norm(joint_accel)**2 * -1e-7
    base_accel = (jnp.linalg.norm(base_lin_accel)**2 + jnp.linalg.norm(base_ang_accel)**2) * -1e-4
    joint_pos_limit = jnp.sum(jnp.where(joint_pos > max_q, 1, 0) + jnp.where(joint_pos < min_q, 1, 0)) * -1
    # jax.debug.print("body pos error {}", jnp.linalg.norm(body_pos_error))
    # jax.debug.print("goal {}", goal_body_vel @ R_b_goal)
    # jax.debug.print("goal leg pos {}", ((goal_joint_body_pos - goal_body_pos) @ R_b_goal)[22])
    # jax.debug.print("leg pos {}", ((joint_body_pos - body_pos) @ R_b)[22])
    # jax.debug.print("body ang vel error{}", joint_body_ang_error)
    # jax.debug.print("body pos error {}", jnp.exp(-jnp.linalg.norm(body_pos_error) / 0.25))
    # jax.debug.print("body vel error {}", jnp.exp(-jnp.linalg.norm(body_vel_error) / 0.25))
    # jax.debug.print("joint body pos error {}", jnp.exp(-jnp.linalg.norm(joint_body_pos_error) / 0.25))
    # jax.debug.print("joint body vel error {}", jnp.exp(-jnp.linalg.norm(joint_body_vel_error) / 0.25))
    # jax.debug.print("body ang error {}", jnp.exp(-jnp.linalg.norm(body_ang_error) / .25))
    # jax.debug.print("body ang vel error {}", jnp.exp(-jnp.linalg.norm(body_ang_vel_error) / 0.25))
    # reward = (survival + errors + 
    #           torque + torque_tiredness + power + joint_accel_reward +
    #           base_accel + joint_pos_limit)

    reward = (errors)
    
    # jax.debug.print("body height{}", body_pos[2])
    reward = jnp.maximum(reward, 0)
            
    # fallen = ((jnp.linalg.norm(body_pos_error) > .4) | (body_pos[2] < .45))
    fallen = ((jnp.linalg.norm(body_pos_error) > .4) | (body_pos[2] < .63))

    end = (fallen) | (step_num > sim.cfg["PPO"]["max_timesteps"])
    reset_anker = (i == 0)

    done = jnp.where(reset_anker, 2, 0)
    done = jnp.where(end, 1, 0)

    # get obs (no push torque)
    obs = jnp.concatenate([body_pos_error, goal_joint_pos, goal_joint_vel, projected_gravity, base_ang_vel, joint_pos, joint_vel, prev_action,
          body_vel, jnp.array((body_pos[2],)), push_force], axis=-1)
    
    return obs, reward, done
