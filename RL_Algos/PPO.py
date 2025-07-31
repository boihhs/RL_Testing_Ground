import jax.numpy as jnp, jax
from jax import random
from jax import lax
from jax.tree_util import register_pytree_node_class
from functools import partial
from Mujoco_Env.Sim import Sim, ENVS
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

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

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

        self.env = Sim(self.cfg)

        self.key, subkey = jax.random.split(self.key)
        value_1 = Value(jnp.array(self.cfg["PPO"]["value_model_shape"]), subkey)
        
        self.key, subkey = jax.random.split(self.key)
        policy = Policy(jnp.array(self.cfg["PPO"]["policy_model_shape"]), jnp.array(self.cfg["PPO"]["default_qpos"]), subkey)
        
        self.buffer = ReplayBuffer(self.cfg["PPO"]["horizon_length"] * (self.cfg["PPO"]["batch_size"]), self.cfg["PPO"]["value_state_dim"], self.cfg["PPO"]["action_dim"], self.cfg["PPO"]["mini_batch_size"])

        learning_rate = float(self.cfg["PPO"]["learning_rate_policy"])
        
        # lr_schedule_value = optax.constant_schedule(
        #     init_value=float(self.cfg["PPO"]["learning_rate_value"])
        # )    

        policy_opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=learning_rate,
            )
        )
        policy_opt_state = policy_opt.init(policy)
        self.policy_container = ModelContainer(policy, policy_opt, policy_opt_state)

        value_1_opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=learning_rate,
            )
        )
        value_1_opt_state = value_1_opt.init(value_1)
        self.value_1_container = ModelContainer(value_1, value_1_opt, value_1_opt_state)

        
    @jax.jit
    def loss(self, buffer, value_1, policy, old_log_probs, old_means, old_log_std):
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

        std     = jnp.exp(log_std)
        old_std = jnp.exp(old_log_std)
        kl = jnp.sum( log_std - old_log_std + 0.5 * ((old_std**2 + (means - old_means)**2) / (std**2)) - 0.5, axis=-1)
        kl_mean = jnp.mean(kl)
        kl_mean = jax.lax.stop_gradient(kl_mean)

        jax.debug.print("dones {}", jnp.mean(dones))
        # jax.debug.print("KL divergence {}", kl_mean)
        # jax.debug.print("surrogate {}", surrogate.shape)
        jax.debug.print("mean mean {}", jnp.mean(jnp.abs(means)))
        jax.debug.print("std mean? {}", jnp.mean(jnp.exp(log_std)))
        # jax.debug.print("loss_value mean? {}", jnp.mean(loss_value))
        # jax.debug.print("policy_loss mean? {}", jnp.mean(policy_loss))
        # jax.debug.print("bound_loss mean? {}", jnp.mean(bound_loss))
        # jax.debug.print("entropy_loss mean? {}", jnp.mean(entropy_loss))
        jax.debug.print("value mean {}", jnp.mean(values))
        
        return loss, kl_mean
        

    def run(self):

        avg_loss = []
        avg_buffer_rewards = []

        key = jax.random.PRNGKey(0)

        key, subkey = jax.random.split(key)

        envs = self.env.reset(subkey)

        @jax.jit
        def _rollout(context, xs):
            envs, buffer, policy, key = context

            key, subkey = jax.random.split(key)
            current_env_obs, rewards, dones = self.env.getObs_and_reward(envs, subkey)
            policy_obs = current_env_obs[:, :self.cfg["PPO"]["policy_state_dim"]]
            value_obs = current_env_obs[:, :self.cfg["PPO"]["value_state_dim"]]

            key, subkey = jax.random.split(key)
            actions = policy.get_action(policy_obs, subkey)
            
            envs = self.env.reset_partial(envs, dones)

            next_envs = self.env.step(envs, actions)

            key, subkey = jax.random.split(key)
            next_env_obs, _, _ = self.env.getObs_and_reward(next_envs, subkey)

            next_value_obs = next_env_obs[:, :self.cfg["PPO"]["value_state_dim"]]
            buffer = buffer.add_batch_PPO(value_obs, actions, rewards, next_value_obs, dones)
            buffer = jax.tree_util.tree_map(jax.lax.stop_gradient, buffer)
            
            return (next_envs, buffer, policy, key), None
        

        @jax.jit
        def _loop_minibatch(context, xs):
            value_1_container, policy_container, buffer, old_log_probs, old_means, old_log_std = context
            value_1, policy = value_1_container.model, policy_container.model
            value_1_opt, policy_opt = value_1_container.opt, policy_container.opt
            value_1_opt_state, policy_opt_state = value_1_container.opt_state, policy_container.opt_state

            (loss, kl_mean), (grads_value_1, grads_policy) = jax.value_and_grad(self.loss, argnums=(1,2), has_aux=True) (buffer, value_1, policy, old_log_probs, old_means, old_log_std)
            
            clip_state_value, inject_state_value = value_1_opt_state
            clip_state_policy, inject_state_policy = policy_opt_state

            learning_rate = inject_state_value.hyperparams['learning_rate']
            learning_rate = jnp.where(kl_mean > self.cfg["PPO"]["desired_kl"] * 2,
                                      jnp.maximum(1e-4, learning_rate / 2),
                                      jnp.where(kl_mean < self.cfg["PPO"]["desired_kl"] / 2, jnp.minimum(1e-2, learning_rate * 1.5), learning_rate))
            
            new_hparams_value  = {**inject_state_value.hyperparams,  'learning_rate': learning_rate}
            new_hparams_policy  = {**inject_state_policy.hyperparams, 'learning_rate': learning_rate}

            inject_state_value  = inject_state_value._replace(hyperparams=new_hparams_value)
            inject_state_policy = inject_state_policy._replace(hyperparams=new_hparams_policy)

            value_1_opt_state = (clip_state_value, inject_state_value)
            policy_opt_state = (clip_state_policy, inject_state_policy)

            updates_value_1, value_1_opt_state = value_1_opt.update(grads_value_1, value_1_opt_state, value_1)
            value_1 = optax.apply_updates(value_1, updates_value_1)
            value_1_container = ModelContainer(value_1, value_1_opt, value_1_opt_state)


            updates_policy, policy_opt_state = policy_opt.update(grads_policy, policy_opt_state, policy)
            policy = optax.apply_updates(policy, updates_policy)
            policy_container = ModelContainer(policy, policy_opt, policy_opt_state)

            jax.debug.print("learning rate: {}", learning_rate)
            return (value_1_container, policy_container, buffer, old_log_probs, old_means, old_log_std), (loss)
        
        for i in range(self.cfg["PPO"]["num_epocs"]):
            policy = self.policy_container.model

            key, subkey = jax.random.split(key)
            
            (envs, self.buffer, _, key), _ = jax.lax.scan(_rollout, (envs, self.buffer, policy, subkey), None, length = int(self.cfg["PPO"]["horizon_length"]))
            
            policy_obs = self.buffer.states[:, :self.cfg["PPO"]["policy_state_dim"]]
            old_log_probs, old_means, old_log_std = policy.get_log_prob(policy_obs, self.buffer.actions)
            old_log_probs = jax.lax.stop_gradient(old_log_probs)
            old_means     = jax.lax.stop_gradient(old_means)
            old_log_std   = jax.lax.stop_gradient(old_log_std)

            (self.value_1_container, self.policy_container, self.buffer, _, _, _), (loss) = jax.lax.scan(_loop_minibatch, (self.value_1_container, self.policy_container, self.buffer, old_log_probs, old_means, old_log_std), None, length = int(self.cfg["PPO"]["mini_batch_loops"]))

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
    

    
        







            



