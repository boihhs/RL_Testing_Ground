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
policy = Policy(jnp.array(cfg["PPO"]["policy_model_shape"]), jnp.array(cfg["PPO"]["action_scale"]), jnp.array(cfg["PPO"]["default_qpos"]), subkey)

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

right_foot_pos_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")
left_foot_pos_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")


def quat_to_small_euler(q):
    # assumes w,x,y,z order; change if yours differs
    qw, qx, qy, qz = q
    pitch = jnp.arcsin(jnp.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
    roll  = jnp.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    # yaw is irrelevant for standing
    return pitch, roll
i = 0
with viewer.launch_passive(mj_model, mj_data) as v:
    while v.is_running():
        frame_start = time.time()

        body_pos = mj_data.qpos[:3]
        quat = mj_data.qpos[3:7]
        body_vel = mj_data.qvel[:3]
        joint_pos = mj_data.qpos[7:]
        joint_vel = mj_data.qvel[6:]
        base_ang_vel = mj_data.sensordata[ang_vel_sensor_start:ang_vel_sensor_start + ang_vel_sensor_length]
        base_lin_accel = mj_data.sensordata[lin_accel_sensor_start:lin_accel_sensor_start + lin_accel_sensor_length]
        right_foot_pos = mj_data.xpos[right_foot_pos_id]
        left_foot_pos = mj_data.xpos[left_foot_pos_id]

        prev_ctrl = mj_data.ctrl[:]

        key, subkey = jax.random.split(key)
        noise_joint_pos = 0 * jax.random.normal(subkey, joint_pos.shape)
        joint_pos = joint_pos + noise_joint_pos

        key, subkey = jax.random.split(key)
        noise_joint_vel = 0 * jax.random.normal(subkey, joint_vel.shape)
        joint_vel = joint_vel + noise_joint_vel

        key, subkey = jax.random.split(key)
        noise_ang_vel = 0 * jax.random.normal(subkey, base_ang_vel.shape)
        base_ang_vel = base_ang_vel + noise_ang_vel

        key, subkey = jax.random.split(key)
        noise_lin_accel = 0 * jax.random.normal(subkey, base_lin_accel.shape)
        base_lin_accel = base_lin_accel + noise_lin_accel

        obs = jnp.concatenate([prev_ctrl, joint_pos, joint_vel, base_ang_vel, base_lin_accel, quat, body_pos, body_vel, right_foot_pos, left_foot_pos], axis=-1)
        # 2. policy → control
        key, subkey = random.split(key)
        actions= policy.get_raw_action(obs[None, :])

        action = actions[0]

        current_ctrl = obs[:23]
        joint_pos = obs[23:46]
        joint_vel = obs[46:69]
        base_ang_vel = obs[69:72]
        base_lin_accel = obs[72:75]
        quat = obs[75:79]
        body_pos = obs[79:82]
        body_vel = obs[82:85]
        right_foot_pos = obs[85:88]
        left_foot_pos = obs[88:91]

        pitch, roll = quat_to_small_euler(quat)                       # torso tilt (rad)

        target   = jnp.array([0, 0.0, 0.65])
        dist     = jnp.linalg.norm(target - body_pos)
        reward_dist   = jnp.exp(-5*dist)

        reward = (
            reward_dist
        )
        reward = jnp.clip(reward, -1.0, 5.0)

        # ---------- termination ----------
        # fallen = (
        #     (jnp.abs(pitch) > jnp.deg2rad(10)) | (jnp.abs(roll)  > jnp.deg2rad(10))
        
        # )
        done = i > 300
        
        print(reward)
        print(done)
        

        # print(action - jnp.array(cfg["PPO"]["default_qpos"]))

        # print(action)
        i = i + 1
        # print(i)
        # print(mj_data.ctrl[:])
        # 3. physics stepping until next control tick
        sim_t0 = mj_data.time
        while (mj_data.time - sim_t0) < DT_CONTROL:

            joint_pos = mj_data.qpos[7:]
            joint_vel = mj_data.qvel[6:]
            ctrl = jnp.array(cfg["PPO"]["stiffness"]) * (action - joint_pos) - jnp.array(cfg["PPO"]["damping"]) * (joint_vel)
            # print(ctrl)
            ctrl.clip(-jnp.array(cfg["PPO"]["torque_limit"]), jnp.array(cfg["PPO"]["torque_limit"]))
            mj_data.ctrl[:] = np.asarray(ctrl, dtype=np.float64) 

            mujoco.mj_step(mj_model, mj_data)

        # render frame
        v.sync()

        # real‑time pacing
        sleep_t = DT_CONTROL - (time.time() - frame_start)
        if sleep_t > 0:
            time.sleep(sleep_t)

        # auto‑reset
        if (done):
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, kf_id)
            episode_start = time.time()
            i = 0


