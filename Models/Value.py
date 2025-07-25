import jax.numpy as jnp, jax
from jax import random
from jax.tree_util import register_pytree_node_class
from functools import partial


@register_pytree_node_class
class Value:
    def __init__(self, layer_sizes, key=random.PRNGKey(0)):
        self.params = self.init_network_params(layer_sizes, key)
       
    @staticmethod
    def random_layer_params(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
    
    @staticmethod
    def init_network_params(sizes, key):
        keys = random.split(key, len(sizes))
        return [Value.random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

    @staticmethod
    @jax.jit
    def relu(x):
        return jnp.maximum(0, x)
            
    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=(0))
    def __call__(self, state):
        activations = state
        for w, b in self.params[:-1]:
            outputs = jnp.dot(w, activations) + b
            activations = self.relu(outputs)

        final_w, final_b = self.params[-1]
        logits = jnp.dot(final_w, activations) + final_b

        return logits

    def tree_flatten(self):
        children = (self.params,)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        
        obj = cls.__new__(cls)
        obj.params, = children
        return obj