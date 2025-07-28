"""
Run a trained PPO policy in MuJoCo viewer at 60 FPS.
"""

import time, re, threading
from pathlib import Path
from Models.Policy import Policy
import mujoco
from mujoco import viewer
from pynput import keyboard
import yaml
import numpy as np
import jax, jax.numpy as jnp
from flax.training import checkpoints
from jax import random
from mujoco import mjx

def print_pytree_structure(pytree, indent=0, path=""):
    if isinstance(pytree, dict):
        for key, value in pytree.items():
            print("  " * indent + f"{path + str(key)}:")
            print_pytree_structure(value, indent + 1, path + str(key) + ".")
    elif isinstance(pytree, (list, tuple)):
        for i, value in enumerate(pytree):
            print("  " * indent + f"{path}[{i}]:")
            print_pytree_structure(value, indent + 1, path + f"[{i}].")
    else:
        print("  " * indent + f"{path[:-1]}")


cfg_file = "/home/leo-benaharon/Desktop/RL_Testing_Ground/RL_Algos/PPO.yaml"
with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

CKPT_DIR = Path("checkpoints").absolute()
CKPT_PREFIX = "policy_"            # adjust if you used a different prefix
DT_TARGET = 1.0 / 60.0             # 60 FPS

# ── 0. MuJoCo sizes ────────────────────────────────────────────────────────
mj_model = mujoco.MjModel.from_xml_path(cfg["PPO"]["xml_path"])
nq, nv, nu = mj_model.nq, mj_model.nv, mj_model.nu

key = random.PRNGKey(0)
key, subkey = jax.random.split(key)
policy = Policy(jnp.array(cfg["PPO"]["policy_model_shape"]), jnp.array(cfg["PPO"]["default_qpos"]), subkey)

params_template = policy

ckpt_path = checkpoints.latest_checkpoint(CKPT_DIR, prefix=CKPT_PREFIX)
if ckpt_path:
    policy = checkpoints.restore_checkpoint(ckpt_path, target=params_template)
    step   = int(re.search(r"_([0-9]+)$", ckpt_path).group(1))
    print(f"✓ loaded step {step} from {ckpt_path}")
else:
    print("[WARN] no checkpoint found; using random weights.")
    policy = params_template





# ── 4. Keys for manual overrides (optional) ────────────────────────────────
pressed_keys = set()
def on_press(key):
    try:    pressed_keys.add(key.char)
    except AttributeError: pressed_keys.add(str(key))
def on_release(key):
    try:    pressed_keys.discard(key.char)
    except AttributeError: pressed_keys.discard(str(key))
keyboard.Listener(on_press=on_press, on_release=on_release).start()

# ── 5. Actuator IDs, camera, keyframe reset ────────────────────────────────


kf_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")

mj_data = mujoco.MjData(mj_model)
mujoco.mj_resetDataKeyframe(mj_model, mj_data, kf_id)

mujoco.mj_resetDataKeyframe(mj_model, mj_data, kf_id)

episode_start = time.time()
DT_CONTROL = 1.0 / cfg["PPO"]["model_freq"]


ang_vel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
ang_vel_sensor_start, ang_vel_sensor_length = mj_model.sensor_adr[ang_vel_id], mj_model.sensor_dim[ang_vel_id]

lin_accel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "linear-acceleration")
lin_accel_sensor_start, lin_accel_sensor_length = mj_model.sensor_adr[lin_accel_id], mj_model.sensor_dim[lin_accel_id]

right_foot_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")
left_foot_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")

body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")

right_foot_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom")
left_foot_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom")
ground_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

def get_collision(d, geom1: int, geom2: int):
    contact = d.contact
    if contact.geom.shape[0] > 0:
        mask = (jnp.array([geom1, geom2]) == contact.geom).all(axis=1)
        mask |= (jnp.array([geom2, geom1]) == contact.geom).all(axis=1)
        idx = jnp.where(mask, contact.dist, 1e4).argmin()
        dist = contact.dist[idx] * mask[idx]
        return dist < 0
    else:
        return False


mjx_data = mjx.put_data(mj_model, mj_data)


def _quat_to_small_euler(q):
    qw, qx, qy, qz = q
    # Yaw (around z-axis)
    yaw = jnp.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    # Pitch (around y-axis)
    pitch = jnp.arcsin(jnp.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
    # Roll (around x-axis)
    roll = jnp.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    return pitch, roll, yaw


def mjcb_control(model, data):
    
    if not hasattr(mjcb_control, "step"):
        mjcb_control.step = 0                                   # step counter
        mjcb_control.last_action = np.zeros(model.nu)           # cached policy output
        # Compute how many sim-steps to skip between policy calls
        mjcb_control.policy_interval = int(1.0 / (model.opt.timestep * 50.0))
        mjcb_control.key = random.PRNGKey(0)
        mjcb_control.i = 0


    d = data
    if mjcb_control.step % mjcb_control.policy_interval == 0:

        mjcb_control.key, subkey = jax.random.split(mjcb_control.key)
        force = jax.random.normal(subkey,  mjx_data.xfrc_applied[body_id][3:].shape) * cfg["STD"]["std_force"]
        mjcb_control.xfrc_applied = jnp.zeros(mjx_data.xfrc_applied[body_id].shape).at[3:].set(force)
        
        body_pos = d.qpos[:3]
        quat = d.qpos[3:7]
        body_vel = d.qvel[:3]
        joint_pos = d.qpos[7:]
        joint_vel = d.qvel[6:]

        base_ang_vel = d.sensordata[ang_vel_sensor_start:ang_vel_sensor_start + ang_vel_sensor_length]
        base_lin_accel = d.sensordata[lin_accel_sensor_start:lin_accel_sensor_start + lin_accel_sensor_length]

        right_foot_pos = d.xpos[right_foot_body_id]
        right_foot_vel = d.cvel[right_foot_body_id][3:]
        right_foot_ang_vel = d.cvel[right_foot_body_id][:3]
        right_foot_force = d.cfrc_ext[right_foot_body_id][3:]
        right_foot_moment = d.cfrc_ext[right_foot_body_id][:3]
        right_foot_ground_contact = get_collision(d, right_foot_geom_id, ground_geom_id)

        left_foot_pos = d.xpos[left_foot_body_id]
        left_foot_vel = d.cvel[left_foot_body_id][3:]
        left_foot_ang_vel = d.cvel[left_foot_body_id][:3]
        left_foot_force = d.cfrc_ext[left_foot_body_id][3:]
        left_foot_ang_moment = d.cfrc_ext[left_foot_body_id][:3]
        left_foot_ground_contact = get_collision(d, left_foot_geom_id, ground_geom_id)

        prev_ctrl = mj_data.ctrl[:]

        obs = jnp.concatenate([
            prev_ctrl, joint_pos, joint_vel, base_ang_vel, base_lin_accel, quat, 
            body_pos, body_vel,
            right_foot_pos, left_foot_pos,
            right_foot_vel, left_foot_vel,
            right_foot_ang_vel, left_foot_ang_vel,
            right_foot_force, left_foot_force,
            right_foot_moment, left_foot_ang_moment,
            jnp.array([right_foot_ground_contact, left_foot_ground_contact])
        ], axis=-1)

        policy_obs = obs[:cfg["PPO"]["policy_state_dim"]]

        actions= policy.get_raw_action(policy_obs[None, :])
        
        # key, subkey = jax.random.split(key)
        # actions= policy.get_action(policy_obs[None, :], subkey)

        mjcb_control.last_action = actions[0]

        current_ctrl           = obs[:23]
        joint_pos              = obs[23:46]
        joint_vel              = obs[46:69]
        base_ang_vel           = obs[69:72]
        base_lin_accel         = obs[72:75]
        quat                   = obs[75:79]
        body_pos               = obs[79:82]
        body_vel               = obs[82:85]
        right_foot_pos         = obs[85:88]
        left_foot_pos          = obs[88:91]
        right_foot_vel         = obs[91:94]
        left_foot_vel          = obs[94:97]
        right_foot_ang_vel     = obs[97:100]
        left_foot_ang_vel      = obs[100:103]
        right_foot_force       = obs[103:106]
        left_foot_force        = obs[106:109]
        right_foot_moment      = obs[109:112]
        left_foot_ang_moment   = obs[112:115]
        right_foot_ground_contact = obs[115]
        left_foot_ground_contact  = obs[116]
        
        pitch, roll, yaw = _quat_to_small_euler(quat)

        z_height_reward = jnp.exp(-10 * (body_pos[2] - 0.65)**2)
        joint_vel_penalty = jnp.exp(-1e-1 * jnp.linalg.norm(joint_vel)**2)
        base_vel_penalty = jnp.exp(-2 * jnp.linalg.norm(body_vel)**2)
        upright_reward = jnp.exp(-10 * (pitch**2 + roll**2))

        reward = (
            2.0 * upright_reward +
            2.0 * z_height_reward +
            1.0 * base_vel_penalty +
            0.5 * joint_vel_penalty
        )

        # ---------- termination conditions -----------------------------------
        too_tilt = (jnp.abs(pitch) > jnp.deg2rad(30)) | (jnp.abs(roll) > jnp.deg2rad(30))
  
        
        # feet_too_far = 
        done = jnp.where(
            too_tilt,
            1,
            0,
        )
    
        # print(reward)
        # print(done)
        # print(data.xfrc_applied[body_id][3:])
        mjcb_control.i = mjcb_control.i + 1
        print(mjcb_control.i)
        
    joint_pos = mj_data.qpos[7:]
    joint_vel = mj_data.qvel[6:]
    ctrl = jnp.array(cfg["PPO"]["stiffness"]) * (mjcb_control.last_action - joint_pos) - jnp.array(cfg["PPO"]["damping"]) * (joint_vel)

    ctrl = ctrl.clip(-jnp.array(cfg["PPO"]["torque_limit"]), jnp.array(cfg["PPO"]["torque_limit"]))
    # print(ctrl)
    mj_data.ctrl[:] = np.asarray(ctrl, dtype=np.float64) 
    mj_data.xfrc_applied[body_id] = mjcb_control.xfrc_applied

    

    

# mj_data.qvel[:3] = np.array([3, 0, 0])
mujoco.set_mjcb_control(mjcb_control)
viewer.launch(mj_model, mj_data)


