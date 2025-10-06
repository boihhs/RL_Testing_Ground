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
from torch.utils.tensorboard import SummaryWriter
from flax.core import FrozenDict
writer = SummaryWriter(log_dir="./logs/ppo")


np.set_printoptions(threshold=np.inf, linewidth=np.inf)

@jax.tree_util.register_dataclass
@dataclass
class ModelContainer:
    module: any = field(metadata={"static": True})
    params: any
    opt: optax.GradientTransformation = field(metadata={"static": True})
    opt_state: optax.OptState

@register_pytree_node_class
class PPO:
    def __init__(self, cfg_file):

        self.key = random.PRNGKey(0)

        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.env = Sim(self.cfg)
        
        self.buffer = ReplayBuffer(self.cfg["PPO"]["horizon_length"] * (self.cfg["PPO"]["batch_size"]), self.cfg["PPO"]["value_state_dim"], self.cfg["PPO"]["action_dim"], self.cfg["PPO"]["mini_batch_size"])

        learning_rate = float(self.cfg["PPO"]["learning_rate_policy"])
        
        # lr_schedule_value = optax.constant_schedule(
        #     init_value=float(self.cfg["PPO"]["learning_rate_value"])
        # )

        policy_module = Policy(
            layer_sizes=tuple(self.cfg["PPO"]["policy_model_shape"]),
            action_bias=jnp.array(self.cfg["PPO"]["default_qpos"])
        )
        value_module = Value(layer_sizes=tuple(self.cfg["PPO"]["value_model_shape"]))


        policy_opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=learning_rate,
            )
        )
        value_1_opt = optax.chain( 
            optax.clip_by_global_norm(0.5), 
            optax.inject_hyperparams(optax.adam)(
                learning_rate=learning_rate, 
                ) 
        )

        self.key, subkey = jax.random.split(self.key)
        value_params = value_module.init(subkey, jnp.ones((1, self.cfg["PPO"]["value_state_dim"])))

        self.key, subkey = jax.random.split(self.key)
        policy_params = policy_module.init(subkey, jnp.ones((1, self.cfg["PPO"]["policy_state_dim"])))

        policy_opt_state = policy_opt.init(policy_params)
        self.policy_container = ModelContainer(policy_module, policy_params, policy_opt, policy_opt_state)

        value_1_opt_state = value_1_opt.init(value_params)
        self.value_1_container = ModelContainer(value_module, value_params, value_1_opt, value_1_opt_state)


        
    @jax.jit
    def loss(self, buffer, value_params, policy_params, old_log_probs, old_means, old_log_std):
        states, actions, rewards, next_states, dones = buffer.states, buffer.actions, buffer.rewards, buffer.next_states, buffer.dones

        values      = self.value_1_container.module.apply(value_params, states)
        next_values = self.value_1_container.module.apply(value_params, next_states)

        deltas = rewards[:, None] + self.cfg["PPO"]["gamma"] * (dones == 0)[:, None] * next_values - values
        deltas_batch = deltas.reshape((self.cfg["PPO"]["batch_size"], self.cfg["PPO"]["horizon_length"]))
        dones_batch  = dones.reshape((self.cfg["PPO"]["batch_size"], self.cfg["PPO"]["horizon_length"]))

        @jax.vmap 
        def _calc_all(deltas, dones):
            T = deltas.shape[0]
            def _calc_adv(context, xs):
                advantage = context
                delta, done = xs
                advantage = delta + self.cfg["PPO"]["gamma"] * self.cfg["PPO"]["lambda"] * (done == 0) * advantage
                return advantage, advantage
            advantage = jnp.zeros(deltas[0].shape)
            _, advantages = jax.lax.scan(_calc_adv, advantage, (deltas, dones), length=T, reverse=True)
            return advantages

        advantages_batch = _calc_all(deltas_batch, dones_batch)
        advantages = advantages_batch.reshape(deltas.shape)

        returns = values + advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = jax.tree_util.tree_map(jax.lax.stop_gradient, advantages)
        returns    = jax.tree_util.tree_map(jax.lax.stop_gradient, returns)

        loss_value = jnp.mean((returns - values) ** 2)

        policy_obs = states[:, :self.cfg["PPO"]["policy_state_dim"]]
        new_log_probs, means, log_std = self.policy_container.module.get_log_prob(policy_params, policy_obs, actions)

        ratio = jnp.exp(new_log_probs - old_log_probs)
        surrogate = advantages * ratio[:, None]
        surrogate_clipped = advantages * jnp.clip(ratio, 1.0 - self.cfg["PPO"]["e_clip"], 1.0 + self.cfg["PPO"]["e_clip"])[:, None]
        policy_loss = -jnp.mean(jnp.minimum(surrogate, surrogate_clipped))

        bound_loss = jnp.mean(jnp.clip(means - 1.0, 0.0, jnp.inf)**2) + jnp.mean(jnp.clip(-(means + 1.0), 0.0, jnp.inf)**2)

        entropy_loss = jnp.mean(.5 * jnp.log(2*jnp.pi*jnp.exp(1)) + log_std)
        total_loss = loss_value + policy_loss + float(self.cfg["PPO"]["bound_coef"]) * bound_loss + float(self.cfg["PPO"]["entropy_coef"]) * entropy_loss

        std     = jnp.exp(log_std)
        old_std = jnp.exp(old_log_std)
        kl = jnp.sum( log_std - old_log_std + 0.5 * ((old_std**2 + (means - old_means)**2) / (std**2)) - 0.5, axis=-1)
        kl_mean = jnp.mean(kl)
        kl_mean = jax.lax.stop_gradient(kl_mean)
        
        stats = FrozenDict({
            "dones_mean": jnp.mean(jnp.where(dones > 0, dones, 0)),
            "mean_abs_mean": jnp.mean(jnp.abs(means)),
            "std_exp_mean": jnp.mean(jnp.exp(log_std)),
            "value_mean": jnp.mean(values),
            "kl_mean"   : kl_mean,
            "policy_loss": policy_loss,
            "value loss": loss_value,
            "entropy_loss": entropy_loss * float(self.cfg["PPO"]["entropy_coef"]),
            "bound_loss": bound_loss * float(self.cfg["PPO"]["bound_coef"]),
            "total_loss": total_loss
        })
        return total_loss, stats
        

    def run(self):

        avg_loss = []
        avg_buffer_rewards = []

        key = jax.random.PRNGKey(0)

        key, subkey = jax.random.split(key)

        envs = self.env.reset(subkey)

        @jax.jit
        def _rollout(context, xs):
            envs, buffer, policy_params, key = context

            key, subkey = jax.random.split(key)
            current_env_obs, rewards, dones, reward_terms = self.env.getObs_and_reward(envs, subkey)
            policy_obs = current_env_obs[:, :self.cfg["PPO"]["policy_state_dim"]]
            value_obs  = current_env_obs[:, :self.cfg["PPO"]["value_state_dim"]]

            key, subkey = jax.random.split(key)
            actions = self.policy_container.module.get_action(policy_params, policy_obs, subkey)

            envs = self.env.reset_partial(envs, dones)
            next_envs = self.env.step(envs, actions)

            key, subkey = jax.random.split(key)
            next_env_obs, _, _, _ = self.env.getObs_and_reward(next_envs, subkey)
            next_value_obs = next_env_obs[:, :self.cfg["PPO"]["value_state_dim"]]

            buffer = buffer.add_batch_PPO(value_obs, actions, rewards, next_value_obs, dones)
            buffer = jax.tree_util.tree_map(jax.lax.stop_gradient, buffer)

            return (next_envs, buffer, policy_params, key), reward_terms
        

        @jax.jit
        def _loop_minibatch(context, xs):
            value_1_container, policy_container, buffer, old_log_probs, old_means, old_log_std = context
            value_params, policy_params = value_1_container.params, policy_container.params
            value_opt, policy_opt = value_1_container.opt, policy_container.opt
            value_opt_state, policy_opt_state = value_1_container.opt_state, policy_container.opt_state

            (loss, stats), (grads_value, grads_policy) = jax.value_and_grad(
                self.loss, argnums=(1, 2), has_aux=True
            )(buffer, value_params, policy_params, old_log_probs, old_means, old_log_std)

            kl_mean = stats["kl_mean"]
            clip_state_value, inject_state_value   = value_opt_state
            clip_state_policy, inject_state_policy = policy_opt_state

            learning_rate = inject_state_value.hyperparams['learning_rate']
            learning_rate = jnp.where(
                kl_mean > self.cfg["PPO"]["desired_kl"] * 2,
                jnp.maximum(1e-4, learning_rate / 2),
                jnp.where(
                    kl_mean < self.cfg["PPO"]["desired_kl"] / 2,
                    jnp.minimum(1e-2, learning_rate * 1.5),
                    learning_rate
                )
            )
            new_hparams_value  = {**inject_state_value.hyperparams,  'learning_rate': learning_rate}
            new_hparams_policy = {**inject_state_policy.hyperparams, 'learning_rate': learning_rate}

            inject_state_value  = inject_state_value._replace(hyperparams=new_hparams_value)
            inject_state_policy = inject_state_policy._replace(hyperparams=new_hparams_policy)

            value_opt_state  = (clip_state_value,  inject_state_value)
            policy_opt_state = (clip_state_policy, inject_state_policy)

            updates_value,  value_opt_state  = value_opt.update(grads_value,  value_opt_state,  value_params)
            value_params = optax.apply_updates(value_params, updates_value)

            updates_policy, policy_opt_state = policy_opt.update(grads_policy, policy_opt_state, policy_params)
            policy_params = optax.apply_updates(policy_params, updates_policy)

            value_1_container = ModelContainer(value_1_container.module, value_params, value_opt, value_opt_state)
            policy_container  = ModelContainer(policy_container.module,  policy_params,  policy_opt, policy_opt_state)

            # jax.debug.print("learning rate: {}", learning_rate)
            return (value_1_container, policy_container, buffer, old_log_probs, old_means, old_log_std), (loss, stats)

        
        for i in range(self.cfg["PPO"]["num_epocs"]):
            # policy = self.policy_container.module

            key, subkey = jax.random.split(key)
            
            (envs, self.buffer, self.policy_container.params, key), reward_terms = jax.lax.scan(_rollout, (envs, self.buffer, self.policy_container.params, subkey), None, length=int(self.cfg["PPO"]["horizon_length"]))
            
            policy_obs = self.buffer.states[:, :self.cfg["PPO"]["policy_state_dim"]]
            old_log_probs, old_means, old_log_std = self.policy_container.module.get_log_prob(self.policy_container.params, policy_obs, self.buffer.actions)
            old_log_probs = jax.lax.stop_gradient(old_log_probs)
            old_means     = jax.lax.stop_gradient(old_means)
            old_log_std   = jax.lax.stop_gradient(old_log_std)

            (self.value_1_container, self.policy_container, self.buffer, _, _, _), (loss, stats) = jax.lax.scan(_loop_minibatch, (self.value_1_container, self.policy_container, self.buffer, old_log_probs, old_means, old_log_std), None, length = int(self.cfg["PPO"]["mini_batch_loops"]))

            reward_mean = float(jnp.mean(self.buffer.rewards))
            loss_mean   = float(jnp.mean(loss))

            stats_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x), stats)
            rewards_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x), reward_terms)

            stats_mean  = dict(stats_mean)
            rewards_mean = dict(rewards_mean)

            # Log individual components
            for k, v in stats_mean.items():
                writer.add_scalar(f"stats/{k}", float(v), i)

            for k, v in rewards_mean.items():
                writer.add_scalar(f"rewards/{k}", float(v), i)
           
            avg_buffer_rewards.append(jnp.mean(self.buffer.rewards))
            avg_loss.append(jnp.mean(loss))
            if (i % 100 == 0):
                ckpt_dir = Path("checkpoints").resolve()
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                checkpoints.save_checkpoint(
                    ckpt_dir=ckpt_dir,
                    target={
                        "policy_params": self.policy_container.params,
                        "policy_opt_state": self.policy_container.opt_state,
                    },
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
    

    
        







            



