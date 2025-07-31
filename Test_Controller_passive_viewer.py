"""
Run a trained PPO policy in MuJoCo viewer at 60 FPS.
"""

import time, re, threading
from pathlib import Path
from Models.Policy import Policy
from Mujoco_Env.Sim import ENVS, Sim, MODEL
from Robot_Models.booster_t1.booster import get_obs_and_reward_walking
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

key = random.PRNGKey(8)
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
print(mj_model.body_mass)
mujoco.mj_resetDataKeyframe(mj_model, mj_data, kf_id)

mujoco.mj_resetDataKeyframe(mj_model, mj_data, kf_id)

episode_start = time.time()
DT_CONTROL = 1.0 / cfg["PPO"]["model_freq"]

mjx_data = mjx.put_data(mj_model, mj_data)
curr_action = jnp.array(cfg["PPO"]["default_qpos"])
prev_action = jnp.array(cfg["PPO"]["default_qpos"])
step_num = 0

goal_velicy = jnp.array([0, -.3, 0])

model = MODEL(jnp.array(mj_model.body_mass), None, None, None, None, None)
push_force = jnp.array([0, 0])

env = ENVS(mjx_data, model, curr_action, prev_action, step_num, None, None, push_force, goal_velicy, None)
sim = Sim(cfg)

prev_action = jnp.array(cfg["PPO"]["default_qpos"])
rewards = []
i = 0
with viewer.launch_passive(mj_model, mj_data) as v:
    while v.is_running():
        frame_start = time.time()

        d = mj_data
        
        key, subkey = jax.random.split(key)
        obs, reward, done = get_obs_and_reward_walking(env, sim, subkey)
        

        policy_obs = obs[:cfg["PPO"]["policy_state_dim"]]
       
        actions= policy.get_raw_action(policy_obs[None, :])
        
        # key, subkey = jax.random.split(key)
        # actions= policy.get_action(policy_obs[None, :], subkey)

        action = actions[0]
    
        # print(reward)
        # print(done)
        rewards.append(reward)
    
        # print(data.xfrc_applied[body_id][3:])
        i = i + 1
        # print(action)
        # 3. physics stepping until next control tick
        sim_t0 = mj_data.time
        while (mj_data.time - sim_t0) < DT_CONTROL:

            joint_pos = mj_data.qpos[7:]
            joint_vel = mj_data.qvel[6:]
            ctrl = jnp.array(cfg["PPO"]["stiffness"]) * (action - joint_pos) - jnp.array(cfg["PPO"]["damping"]) * (joint_vel)

            # ctrl = jnp.array(cfg["PPO"]["stiffness"]) * (jnp.array(cfg["PPO"]["default_qpos"]) - joint_pos) - jnp.array(cfg["PPO"]["damping"]) * (joint_vel)
            # print(ctrl)
            ctrl = ctrl.clip(-jnp.array(cfg["PPO"]["torque_limit"]), jnp.array(cfg["PPO"]["torque_limit"]))
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
            print("hello")
            print(np.mean(np.array(rewards)))
            rewards = []
            i = 0
            mjx_data = mjx.put_data(mj_model, mj_data)
            env = ENVS(mjx_data, env.model, jnp.array(cfg["PPO"]["default_qpos"]), jnp.array(cfg["PPO"]["default_qpos"]), 0, None, None, env.force_applied, env.goal_velocity, None)
        else:
            mjx_data = mjx.put_data(mj_model, mj_data)
            env = ENVS(mjx_data, env.model, action, env.curr_action, env.step_num + 1, None, None, env.force_applied, env.goal_velocity, None)

            
