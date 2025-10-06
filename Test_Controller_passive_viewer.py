import time, re
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from jax import random
from flax.training import checkpoints
import mujoco
from mujoco import viewer, mjx
from pynput import keyboard
import yaml

from Models.Policy import Policy
from Mujoco_Env.Sim import ENVS, Sim, MODEL
from Robot_Models.zeroth.zeroth import get_obs_and_reward_walking

class ViewerRunner:
    def __init__(self, cfg_file: str, goal_vel: jnp.array([0, 0, 0]), deterministic: bool = True):
        # --- Load config
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.key = random.PRNGKey(8)
        self.deterministic = deterministic   # <── NEW flag
        self.ckpt_dir = Path("checkpoints").absolute()
        self.ckpt_prefix = "policy_"

        # --- Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(self.cfg["PPO"]["xml_path"])
        self.mj_data = mujoco.MjData(self.mj_model)

        # Reset to keyframe "home"
        self.kf_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.kf_id)

        # --- Wrap in MJX
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        # --- Env wrapper
        self.env = ENVS(
            self.mjx_data,
            MODEL(jnp.array(self.mj_model.body_mass), None, None, None, None, None),
            jnp.array(self.cfg["PPO"]["default_qpos"]),
            jnp.array(self.cfg["PPO"]["default_qpos"]),
            0,
            None,
            None,
            jnp.array([0, 0]),               # force_applied
            goal_vel,         # goal velocity
            None
        )
        self.sim = Sim(self.cfg)

        # --- Build Flax policy module
        self.policy_module = Policy(
            layer_sizes=jnp.array(self.cfg["PPO"]["policy_model_shape"]),
            action_bias=jnp.array(self.cfg["PPO"]["default_qpos"])
        )

        # Init params (dummy obs with correct dim)
        self.key, subkey = jax.random.split(self.key)
        dummy_obs = jnp.ones((self.cfg["PPO"]["policy_state_dim"],))
        self.policy_params = self.policy_module.init(subkey, dummy_obs)

        # Restore checkpoint if available
        ckpt_path = checkpoints.latest_checkpoint(self.ckpt_dir, prefix=self.ckpt_prefix)
        if ckpt_path:
            restored = checkpoints.restore_checkpoint(
                ckpt_path, target={"policy_params": self.policy_params}
            )
            if isinstance(restored, dict) and "policy_params" in restored:
                self.policy_params = restored["policy_params"]
            else:
                self.policy_params = restored
            step = int(re.search(r"_([0-9]+)$", ckpt_path).group(1))
            print(f"✓ Loaded step {step} from {ckpt_path}")
        else:
            print("[WARN] No checkpoint found; using random weights.")

        # --- Key listener
        self.pressed_keys = set()
        def on_press(key):
            try:    self.pressed_keys.add(key.char)
            except AttributeError: self.pressed_keys.add(str(key))
        def on_release(key):
            try:    self.pressed_keys.discard(key.char)
            except AttributeError: self.pressed_keys.discard(str(key))
        keyboard.Listener(on_press=on_press, on_release=on_release).start()

        # --- Control timing
        self.dt_control = 1.0 / self.cfg["PPO"]["model_freq"]

        # Rewards history
        self.rewards = []

    def step_policy(self, obs):
        """Compute policy action from observation."""
        pol_obs = obs[:self.cfg["PPO"]["policy_state_dim"]]

        if self.deterministic:
            # Deterministic evaluation
            action = self.policy_module.get_raw_action(
                self.policy_params, pol_obs[None, :])[0]
        else:
            # Stochastic sampling (adds noise via log_std)
            self.key, subkey = jax.random.split(self.key)
            action = self.policy_module.get_action(
                self.policy_params, pol_obs[None, :], subkey)[0]

        return action

    def control_loop(self, action):
        """Apply PD control and step MuJoCo physics until next control tick."""
        sim_t0 = self.mj_data.time
        while (self.mj_data.time - sim_t0) < self.dt_control:
            joint_pos = self.mj_data.qpos[7:]
            joint_vel = self.mj_data.qvel[6:]
            # action = jnp.array(jnp.array(self.cfg["PPO"]["default_qpos"]))

            xfrc_applied_body = jnp.zeros(self.mj_data.xfrc_applied[self.sim.body_id].shape).at[3:5].set(self.env.force_applied)
            xfrc_applied = jnp.zeros(self.mj_data.xfrc_applied.shape).at[self.sim.body_id].set(xfrc_applied_body)

            ctrl = (
                jnp.array(self.cfg["PPO"]["stiffness"]) * (action - joint_pos)
                - jnp.array(self.cfg["PPO"]["damping"]) * joint_vel
            )
            ctrl = ctrl.clip(
                -jnp.array(self.cfg["PPO"]["torque_limit"]),
                 jnp.array(self.cfg["PPO"]["torque_limit"])
            )
            self.mj_data.ctrl[:] = np.asarray(ctrl, dtype=np.float64)
            self.mj_data.xfrc_applied[:] = np.asarray(xfrc_applied, dtype=np.float64)
            mujoco.mj_step(self.mj_model, self.mj_data)

    def run(self):
        """Main viewer loop."""
        with viewer.launch_passive(self.mj_model, self.mj_data) as v:
            i = 0
            while v.is_running():
                frame_start = time.time()

                self.key, subkey = jax.random.split(self.key)
                obs, reward, done, _ = get_obs_and_reward_walking(self.env, self.sim, subkey)
                print(reward)
                print(self.env.goal_velocity)
                action = self.step_policy(obs)
                self.rewards.append(reward)

                self.control_loop(action)

                v.sync()

                sleep_t = self.dt_control - (time.time() - frame_start)
                if sleep_t > 0:
                    time.sleep(sleep_t)

                if (done == 1):
                    mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.kf_id)
                    episode_start = time.time()
                    print(np.mean(np.array(self.rewards)))
                    self.rewards = []
                    i = 0
                    mjx_data = mjx.put_data(self.mj_model, self.mj_data)
                    self.env = ENVS(mjx_data, self.env.model, jnp.array(self.cfg["PPO"]["default_qpos"]), jnp.array(self.cfg["PPO"]["default_qpos"]), 0, None, None, self.env.force_applied, self.env.goal_velocity, None)
                else:
                    if 'w' in self.pressed_keys:
                        goal_velocity = jnp.array([0, -.1, 0]) + self.env.goal_velocity
                    elif 's' in self.pressed_keys:
                        goal_velocity = jnp.array([0, .1, 0]) + self.env.goal_velocity
                    elif 'a' in self.pressed_keys:
                        goal_velocity = jnp.array([-.1, 0, 0]) + self.env.goal_velocity
                    elif 'd' in self.pressed_keys:
                        goal_velocity = jnp.array([0.1, 0, 0]) + self.env.goal_velocity
                    elif 'e' in self.pressed_keys:
                        goal_velocity = jnp.array([0., 0, -.1]) + self.env.goal_velocity
                    elif 'q' in self.pressed_keys:
                        goal_velocity = jnp.array([0., 0, .1]) + self.env.goal_velocity
                    else:
                        goal_velocity = self.env.goal_velocity

                    self.key, subkey = jax.random.split(self.key)
                    force_activate = jax.random.bernoulli(subkey, .03)
                    self.key, subkey = jax.random.split(self.key)
                    force_applied = jax.random.normal(subkey,  self.mj_data.xfrc_applied[self.sim.body_id][3:5].shape) * self.cfg["STD"]["std_force"] * force_activate * 0
                    
                    mjx_data = mjx.put_data(self.mj_model, self.mj_data)
                    self.env = ENVS(mjx_data, self.env.model, action, self.env.curr_action, self.env.step_num + 1, None, None, force_applied, goal_velocity, None)
                i += 1


if __name__ == "__main__":
    cfg_file = "/home/leo-benaharon/Desktop/RL_Testing_Ground/RL_Algos/PPO.yaml"

    # Run deterministic evaluation
    runner = ViewerRunner(cfg_file, goal_vel=jnp.array([0, 0, 0]), deterministic=True)
    runner.run()
