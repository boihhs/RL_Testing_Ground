import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence
from functools import partial


class Policy(nn.Module):
    layer_sizes: Sequence[int]
    action_bias: jnp.ndarray
    log_std_min: float = -5.0
    log_std_max: float = 0.5

    @nn.compact
    def __call__(self, x):
        for size in self.layer_sizes[:-1]:
            x = nn.Dense(size,
                kernel_init=nn.initializers.normal(1e-3),
                bias_init=nn.initializers.normal(1e-3))(x)
            x = nn.relu(x)

        x = nn.Dense(self.layer_sizes[-1],
            kernel_init=nn.initializers.normal(1e-3),
            bias_init=nn.initializers.normal(1e-3))(x)

        out_len = x.shape[-1]
        mu = x[..., :out_len // 2]
        log_std = x[..., out_len // 2:]

        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (jnp.tanh(log_std) + 1)
        return mu, log_std

    def get_action(self, params, x, key):
        mu, log_std = self.apply(params, x)
        noise = jax.random.normal(key, shape=log_std.shape)
        std = jnp.exp(log_std).clip(1e-3, None)
        return (mu + std * noise) + self.action_bias[None, :]

    def get_raw_action(self, params, x):
        mu, log_std = self.apply(params, x)
        return mu + self.action_bias[None, :]

    def get_log_prob(self, params, x, action):
        mu, log_std = self.apply(params, x)
        std = jnp.exp(log_std).clip(1e-3, None)
        pre_action = (action - self.action_bias[None, :])
        log_density = -0.5 * jnp.sum(((pre_action - mu) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
        return log_density, mu, log_std
