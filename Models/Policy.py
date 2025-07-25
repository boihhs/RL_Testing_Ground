import jax.numpy as jnp, jax
from jax import random
from jax.tree_util import register_pytree_node_class
from functools import partial


@register_pytree_node_class
class Policy:
    def __init__(self, layer_sizes, action_scale, action_bias, key=random.PRNGKey(0)):
        self.params = self.init_network_params(layer_sizes, key)
        self.LOG_STD_MIN = -5
        self.LOG_STD_MAX = .5
        self.action_scale = action_scale
        self.action_bias = action_bias
       
    @staticmethod
    def random_layer_params(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
    
    @staticmethod
    def init_network_params(sizes, key):
        keys = random.split(key, len(sizes))
        return [Policy.random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

    @staticmethod
    @jax.jit
    def relu(x):
        return jnp.maximum(0, x)
            
    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=(0, 0))
    def __call__(self, x): 
        
        activations = x
        for w, b in self.params[:-1]:
            outputs = jnp.dot(w, activations) + b
            activations = self.relu(outputs)

        final_w, final_b = self.params[-1]
        logits = jnp.dot(final_w, activations) + final_b

        out_len = logits.shape[-1]

        mu = logits[:out_len // 2]
        log_std = logits[out_len // 2:]
        log_std = self.LOG_STD_MIN + .5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (jnp.tanh(log_std) + 1)
        # log_std = jnp.clip(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX) 

        return mu, log_std
    
    @jax.jit
    def get_action(self, x, key):
        mu, log_std = self(x)

        noise = jax.random.normal(key, shape = log_std.shape)
        std = jnp.exp(log_std).clip(1e-3, None)
        action = (mu + std * noise) + self.action_bias[None, :]


        return action 
    
    @jax.jit
    def get_raw_action(self, x):
        mu, log_std = self(x)
        action = mu + self.action_bias[None, :]
        return action 
    
    @jax.jit
    def get_log_prob(self, x, action):
        mu, log_std = self(x)

        std = jnp.exp(log_std).clip(1e-3, None)

        pre_action = (action - self.action_bias[None, :])

        log_density = -.5 * jnp.sum(((pre_action - mu) / std)**2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
        
        log_prob = (log_density)
        return log_prob, mu, log_std
    

    def tree_flatten(self):
        children = (self.params,)
        aux = (self.LOG_STD_MIN, self.LOG_STD_MAX, self.action_scale, self.action_bias)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        
        obj = cls.__new__(cls)
        obj.params, = children
        obj.LOG_STD_MIN, obj.LOG_STD_MAX, obj.action_scale, obj.action_bias = aux
        return obj