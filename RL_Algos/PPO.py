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

# np.set_printoptions(threshold=np.inf, linewidth=np.inf)

@jax.tree_util.register_dataclass
@dataclass
class ModelContainer:
    model: any
    opt: optax.GradientTransformation = field(metadata={"static": True})
    opt_state: optax.OptState

@register_pytree_node_class
class PPO:
    def __init__(self, cfg_file):

        self.key = random.PRNGKey(0)

        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        env_cfg = SimCfg(self.cfg["PPO"]["xml_path"], self.cfg["PPO"]["batch_size"], self.cfg["PPO"]["model_freq"], jnp.array(self.cfg["PPO"]["init_pos"]), jnp.array(self.cfg["PPO"]["init_vel"]), 
                         self.cfg["STD"]["std_gyro"], self.cfg["STD"]["std_acc"], self.cfg["STD"]["std_joint_pos"], self.cfg["STD"]["std_joint_vel"], self.cfg["STD"]["std_body_mass"], self.cfg["STD"]["std_body_inertia"], self.cfg["STD"]["std_body_ipos"], self.cfg["STD"]["std_geom_friction"], self.cfg["STD"]["std_dof_armature"], self.cfg["STD"]["std_dof_frictionloss"], self.cfg["STD"]["std_stiffness"], self.cfg["STD"]["std_damping"], self.cfg["STD"]["std_qpos"], self.cfg["STD"]["std_qvel"], self.cfg["STD"]["std_force"],
                         jnp.array(self.cfg["PPO"]["stiffness"]), jnp.array(self.cfg["PPO"]["damping"]), jnp.array(self.cfg["PPO"]["torque_limit"]))
        self.env = Sim(env_cfg)

        self.key, subkey = jax.random.split(self.key)
        value_1 = Value(jnp.array(self.cfg["PPO"]["value_model_shape"]), subkey)
        
        self.key, subkey = jax.random.split(self.key)
        policy = Policy(jnp.array(self.cfg["PPO"]["policy_model_shape"]), jnp.array(self.cfg["PPO"]["default_qpos"]), subkey)
        
        self.buffer = ReplayBuffer(self.cfg["PPO"]["horizon_length"] * (self.cfg["PPO"]["batch_size"]), self.cfg["PPO"]["value_state_dim"], self.cfg["PPO"]["action_dim"], self.cfg["PPO"]["mini_batch_size"])

        transition_steps = int((self.cfg["PPO"]["mini_batch_loops"] * self.cfg["PPO"]["num_epocs"]) // 10)

        lr_schedule_policy = optax.exponential_decay(
            init_value=float(self.cfg["PPO"]["learning_rate_policy"]),
            transition_steps=transition_steps,
            decay_rate=0.99,
            staircase=True
        )
        lr_schedule_value = optax.exponential_decay(
            init_value=float(self.cfg["PPO"]["learning_rate_value"]),
            transition_steps=transition_steps,
            decay_rate=0.99,
            staircase=True
        )

        policy_opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adamw(
                learning_rate=lr_schedule_policy,
                weight_decay=float(self.cfg["PPO"]["weight_decay"])
            )
        )
        policy_opt_state = policy_opt.init(policy)
        self.policy_container = ModelContainer(policy, policy_opt, policy_opt_state)

        value_1_opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adamw(
                learning_rate=lr_schedule_value,
                weight_decay=float(self.cfg["PPO"]["weight_decay"])
            )
        )
        value_1_opt_state = value_1_opt.init(value_1)
        self.value_1_container = ModelContainer(value_1, value_1_opt, value_1_opt_state)

        
    @jax.jit
    def loss(self, buffer, value_1, policy, old_log_probs):
        states, actions, rewards, next_states, dones = buffer.states, buffer.actions, buffer.rewards, buffer.next_states, buffer.dones

        values = value_1(states)
        next_values = value_1(next_states)

        deltas = rewards[:, None] + self.cfg["PPO"]["gamma"] * (dones == 0)[:, None] * next_values - values
        deltas_batch = deltas.reshape((self.cfg["PPO"]["batch_size"], self.cfg["PPO"]["horizon_length"]))
        dones_batch = dones.reshape((self.cfg["PPO"]["batch_size"], self.cfg["PPO"]["horizon_length"]))

        @jax.vmap 
        def _calc_all(deltas, dones):
            T = deltas.shape[0]

            @jax.jit
            def _calc_adv(context, xs):
                advantage = context
                delta, done = xs
                advantage = delta + self.cfg["PPO"]["gamma"] * self.cfg["PPO"]["lambda"] * (done == 0) * advantage

                return advantage, advantage
            
            advantage = jnp.zeros(deltas[0].shape)
            _, advantages = jax.lax.scan(_calc_adv, advantage, (deltas, dones), length = T, reverse=True)
            return advantages
        
        advantages_batch = _calc_all(deltas_batch, dones_batch)
        advantages = advantages_batch.reshape(deltas.shape)

        returns = values + advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages = jax.tree_util.tree_map(jax.lax.stop_gradient, advantages)
        returns    = jax.tree_util.tree_map(jax.lax.stop_gradient, returns)

        loss_value = jnp.mean((returns - values) ** 2)

        policy_obs = states[:, :self.cfg["PPO"]["policy_state_dim"]]
        new_log_probs, means, log_std = policy.get_log_prob(policy_obs, actions)
      
        ratio = jnp.exp(new_log_probs - old_log_probs)
        
        surrogate = advantages * ratio[:, None]
        surrogate_clipped = advantages * jnp.clip(ratio, 1.0 - self.cfg["PPO"]["e_clip"], 1.0 + self.cfg["PPO"]["e_clip"])[:, None]
        policy_loss = -jnp.mean(jnp.minimum(surrogate, surrogate_clipped))
        

        bound_loss = jnp.mean(jnp.clip(means - 1.0, min=0.0)**2) + jnp.mean(jnp.clip(means + 1.0, max=0.0)**2)

        entropy_loss = jnp.mean(.5 * jnp.log(2*jnp.pi*jnp.exp(1)) + log_std)

        loss = loss_value + policy_loss + float(self.cfg["PPO"]["bound_coef"]) * bound_loss + float(self.cfg["PPO"]["entropy_coef"]) * entropy_loss
        # jax.debug.print("dones {}", dones)
        # jax.debug.print("states {}", states[:, 77])
        # jax.debug.print("surrogate {}", surrogate_clipped.transpose())
        jax.debug.print("mean mean {}", jnp.mean(jnp.abs(means)))
        jax.debug.print("std mean? {}", jnp.mean(jnp.exp(log_std)))
        jax.debug.print("loss_value mean? {}", jnp.mean(loss_value))
        jax.debug.print("policy_loss mean? {}", jnp.mean(policy_loss))
        jax.debug.print("bound_loss mean? {}", jnp.mean(bound_loss))
        jax.debug.print("entropy_loss mean? {}", jnp.mean(entropy_loss))
        jax.debug.print("value mean {}", jnp.mean(values))
        
        return loss
        

    def run(self):

        avg_loss = []
        avg_buffer_rewards = []

        key = jax.random.PRNGKey(0)

        key, subkey = jax.random.split(key)

        envs = self.env.reset(subkey)

        @jax.jit
        def _rollout(context, xs):
            envs, buffer, policy, prev_action, key = context

            key, subkey = jax.random.split(key)
            current_env_obs = self.env.getObs(envs, subkey)
            policy_obs = current_env_obs[:, :self.cfg["PPO"]["policy_state_dim"]]
            value_obs = current_env_obs[:, :self.cfg["PPO"]["value_state_dim"]]

            key, subkey = jax.random.split(key)
            actions = policy.get_action(policy_obs, subkey)
            

            rewards, dones = self.get_reward_and_dones(current_env_obs, actions, envs.step_num, prev_action)
            envs = self.env.reset_partial(envs, dones)

            next_envs = self.env.step(envs, actions)

            key, subkey = jax.random.split(key)
            next_env_obs = self.env.getObs(next_envs, subkey)
            next_value_obs = next_env_obs[:, :self.cfg["PPO"]["value_state_dim"]]
            buffer = buffer.add_batch_PPO(value_obs, actions, rewards, next_value_obs, dones)
            buffer = jax.tree_util.tree_map(jax.lax.stop_gradient, buffer)
            
            return (next_envs, buffer, policy, actions, key), None
        

        @jax.jit
        def _loop_minibatch(context, xs):
            value_1_container, policy_container, buffer, old_log_probs = context
            value_1, policy = value_1_container.model, policy_container.model
            value_1_opt, policy_opt = value_1_container.opt, policy_container.opt
            value_1_opt_state, policy_opt_state = value_1_container.opt_state, policy_container.opt_state

            loss, (grads_value_1, grads_policy) = jax.value_and_grad(self.loss, argnums=(1,2)) (buffer, value_1, policy, old_log_probs)
            
            updates_value_1, value_1_opt_state = value_1_opt.update(grads_value_1, value_1_opt_state, value_1)
            value_1 = optax.apply_updates(value_1, updates_value_1)
            value_1_container = ModelContainer(value_1, value_1_opt, value_1_opt_state)


            updates_policy, policy_opt_state = policy_opt.update(grads_policy, policy_opt_state, policy)
            policy = optax.apply_updates(policy, updates_policy)
            policy_container = ModelContainer(policy, policy_opt, policy_opt_state)

            return (value_1_container, policy_container, buffer, old_log_probs), (loss)
        
        for i in range(self.cfg["PPO"]["num_epocs"]):
            policy = self.policy_container.model

            key, subkey = jax.random.split(key)
            prev_action = jnp.broadcast_to(jnp.array(self.cfg["PPO"]["default_qpos"]), (int(self.cfg["PPO"]["batch_size"]), jnp.array(self.cfg["PPO"]["default_qpos"]).shape[0]))
            print(prev_action.shape)
            (envs, self.buffer, _, _, key), _ = jax.lax.scan(_rollout, (envs, self.buffer, policy, prev_action, subkey), None, length = int(self.cfg["PPO"]["horizon_length"]))
            
            policy_obs = self.buffer.states[:, :self.cfg["PPO"]["policy_state_dim"]]
            old_log_probs, _, _ = policy.get_log_prob(policy_obs, self.buffer.actions)
            old_log_probs = jax.tree_util.tree_map(jax.lax.stop_gradient, old_log_probs)

            (self.value_1_container, self.policy_container, self.buffer, _), (loss) = jax.lax.scan(_loop_minibatch, (self.value_1_container, self.policy_container, self.buffer, old_log_probs), None, length = int(self.cfg["PPO"]["mini_batch_loops"]))

            print("Rewards Batch Average:", jnp.mean(self.buffer.rewards), "loss Average:", jnp.mean(loss))
            avg_buffer_rewards.append(jnp.mean(self.buffer.rewards))
            avg_loss.append(jnp.mean(loss))

            ckpt_dir = Path("checkpoints").resolve()
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            checkpoints.save_checkpoint(
                ckpt_dir=ckpt_dir,
                target=self.policy_container.model,
                step=i,
                prefix="policy_",
                overwrite=True
            )

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(avg_buffer_rewards)
        plt.title("avg_buffer_rewards")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(avg_loss)
        plt.title("avg_loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    @jax.jit
    def get_reward_and_dones(self, obs, actions, step_counters, prev_actions):

        @partial(jax.vmap, in_axes=(0, 0, 0, 0), out_axes=(0, 0))
        def _get_reward_and_done_helper(obs, actions, step_counter, prev_action):

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
            time = step_counter * (1 / self.cfg["PPO"]["model_freq"])
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

            joint_pos_upper = jnp.array(self.cfg["PPO"]["joint_q_max"])
            joint_pos_lower = jnp.array(self.cfg["PPO"]["joint_q_min"])
            tau_limit = jnp.array(self.cfg["PPO"]["torque_limit"])
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
            prog_err = body_pos[1] - v_cmd[1] * time          # torso should have advanced
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
            phase              = (2 * jnp.pi * gait_freq * time) % (2 * jnp.pi)
            left_should_swing  = phase < jnp.pi
            right_should_swing = ~left_should_swing

            

            # feet‑swing (+3 when the correct leg is airborne)
            rew_gait  = 3.0 * (
                left_should_swing.astype(jnp.float32)  * left_off.astype(jnp.float32)
                + right_should_swing.astype(jnp.float32) * right_off.astype(jnp.float32)
            )

            # stance‑slip penalty (‑0.1 · ‖v_xy‖² when foot should stick)
            rew_gait += -0.1 * (
                (1.0 - left_off.astype(jnp.float32))  * jnp.sum(left_foot_vel[:2]**2)
                + (1.0 - right_off.astype(jnp.float32)) * jnp.sum(right_foot_vel[:2]**2)
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
            | (jnp.abs(roll)  > jnp.deg2rad(20)) | (step_counter > self.cfg["PPO"]["max_timesteps"])
            )
            done = fail.astype(jnp.float32)


            return reward, done

        return _get_reward_and_done_helper(obs, actions, step_counters, prev_actions)
    

    def tree_flatten(self):
        children = (self.key, self.value_1_container, self.policy_container, self.buffer)
        aux = (self.cfg, self.env)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):

        obj = cls.__new__(cls)
        (obj.key, obj.value_1_container, obj.policy_container, obj.buffer) = children
        (obj.cfg, obj.env) = aux
   
        return obj
    

    
        







            



