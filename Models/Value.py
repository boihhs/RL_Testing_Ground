import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence
from functools import partial

class Value(nn.Module):
    layer_sizes: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for size in self.layer_sizes[:-1]:
            x = nn.relu(nn.Dense(size,
                kernel_init=nn.initializers.normal(1e-2),
                bias_init=nn.initializers.normal(1e-2))(x))
        return nn.Dense(self.layer_sizes[-1],
            kernel_init=nn.initializers.normal(1e-2),
            bias_init=nn.initializers.normal(1e-2))(x)
