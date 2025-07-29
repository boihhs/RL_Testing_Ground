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

def _quat_to_small_euler(q):
    qw, qx, qy, qz = q
    # Yaw (around z-axis)
    yaw = jnp.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    # Pitch (around y-axis)
    pitch = jnp.arcsin(jnp.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
    # Roll (around x-axis)
    roll = jnp.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    return pitch, roll, yaw

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

prev_action = jnp.array(cfg["PPO"]["default_qpos"])

i = 0
with viewer.launch_passive(mj_model, mj_data) as v:
    while v.is_running():
        frame_start = time.time()

        d = mj_data
        # key, subkey = jax.random.split(key)
        # force = jax.random.normal(subkey,  d.xfrc_applied[body_id][3:].shape) * cfg["STD"]["std_force"]
        # xfrc_applied = jnp.zeros(d.xfrc_applied[body_id].shape).at[3:].set(force)
        
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

        gait_frequency = 1.5

        time_in_seconds = i * (1 / cfg["PPO"]["model_freq"])
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

        policy_obs = obs[:cfg["PPO"]["policy_state_dim"]]

        actions= policy.get_raw_action(policy_obs[None, :])
        
        # key, subkey = jax.random.split(key)
        # actions= policy.get_action(policy_obs[None, :], subkey)

        action = actions[0]

        current_action = actions
        prev_action = prev_action
        v_cmd = jnp.array([0, .1, 0])
        time_in_seconds = i * (1 / cfg["PPO"]["model_freq"])
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

        # ------------------------------------------------------------------
        # constants that match Booster‑Gym Table II (feel free to tweak)
        SIGMA_X   = 0.25
        SIGMA_Y   = 0.25
        DES_HEIGHT = 0.65        # torso height set‑point         [m]
        D_REF      = 0.22        # desired foot spacing           [m]
        H_LIFT     = 0.05        # ≥ this ⇒ foot counted “swing”  [m]
        # ------------------------------------------------------------------

        # 1. ---------------------------------------------------------------- TRACKING
        # velocity tracking (two translational + one yaw)
        prog_err = body_pos[1] - v_cmd[1] * time_in_seconds          # torso should have advanced
        rew_track  = (
            jnp.exp(-((v_cmd[0] - body_vel[0])**2) / SIGMA_X)
            + jnp.exp(-((v_cmd[1] - body_vel[1])**2) / SIGMA_Y)
            + 0.5 * jnp.exp(-((v_cmd[2] - base_ang_vel[2])**2) / SIGMA_X)
        ) - 10.0 * prog_err**2 
        # survival bonus + torso–height term
        rew_track += 0.025
        rew_track += -20.0 * (DES_HEIGHT - body_pos[2])**2

        # orientation penalty (use small‑angle pitch/roll)
        pitch, roll, yaw_base = _quat_to_small_euler(quat)
        rew_track += -5.0 * (pitch**2 + roll**2)

        left_off  = 1 - right_foot_ground_contact
        right_off = 1 - left_foot_ground_contact

        stance_mask   = jnp.stack([1.0 - left_off, 1.0 - right_off])          # 1 if foot on ground
        stance_feet   = jnp.stack([left_foot_pos[:2], right_foot_pos[:2]])
        support_cent  = jnp.sum(stance_feet * stance_mask[:, None], axis=0) \
                        / (jnp.sum(stance_mask) + 1e-6)
        rew_track    += -5.0 * jnp.linalg.norm(body_pos[:2] - support_cent)    # COM alignment

        # 2. ---------------------------------------------------------------- GAIT
        phase              = (2 * jnp.pi * gait_freq * time_in_seconds) % (2 * jnp.pi)
        left_should_swing  = phase < jnp.pi
        right_should_swing = ~left_should_swing

        

        # feet‑swing (+3 when the correct leg is airborne)
        rew_gait  = 3.0 * (
            left_should_swing  * left_off
            + right_should_swing * right_off
        )

        # stance‑slip penalty (‑0.1 · ‖v_xy‖² when foot should stick)
        rew_gait += -0.1 * (
            (1.0 - left_off)  * jnp.sum(left_foot_vel[:2]**2)
            + (1.0 - right_off) * jnp.sum(right_foot_vel[:2]**2)
        )

        # foot spacing
        d_feet  = jnp.linalg.norm(right_foot_pos[:2] - left_foot_pos[:2])
        rew_gait += -1.0 * jnp.maximum(D_REF - d_feet, 0.0)

        # 3. -------------------------------------------------------------- REGULARISATION
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

        # joint‑limit penalty
        limit_violation = jnp.sum(
            jnp.where(joint_pos > joint_pos_upper, 1.0, 0.0)
            + jnp.where(joint_pos < joint_pos_lower, 1.0, 0.0)
        )
        rew_reg += -1.0 * limit_violation

        # 4. -------------------------------------------------------------- TOTAL REWARD
        reward = rew_track + rew_gait + rew_reg

        # 5. -------------------------------------------------------------- DONE FLAG
        # robot “fails” if height too low OR torso tilts too far
        fail  = (
            (body_pos[2] < 0.5)
        | (jnp.abs(pitch) > jnp.deg2rad(20))
        | (jnp.abs(roll)  > jnp.deg2rad(20))
        )
        done = fail

        key, subkey = jax.random.split(key)
        force_activate = jax.random.bernoulli(key, .03) * 0
        key, subkey = jax.random.split(key)
        force = jax.random.normal(subkey,  d.xfrc_applied[body_id][3:5].shape) * cfg["STD"]["std_force"]
        force = jnp.where(force > 0,
                  jnp.maximum(force, cfg["STD"]["std_force"] / 2),
                  jnp.minimum(force, -cfg["STD"]["std_force"] / 2))  * force_activate
        xfrc_applied = jnp.zeros(d.xfrc_applied[body_id].shape).at[3:5].set(force)
    
        print(reward)
        print(done)
    
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
            mj_data.xfrc_applied[body_id] = xfrc_applied

            mujoco.mj_step(mj_model, mj_data)

        # render frame
        v.sync()

        prev_action = action

        # real‑time pacing
        sleep_t = DT_CONTROL - (time.time() - frame_start)
        if sleep_t > 0:
            time.sleep(sleep_t)

        # auto‑reset
        if (done):
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, kf_id)
            episode_start = time.time()
            i = 0
            
