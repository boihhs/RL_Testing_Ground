import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax import random

@register_pytree_node_class
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, batch_size: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = jnp.zeros((capacity, state_dim))
        self.actions = jnp.zeros((capacity, action_dim))
        self.rewards = jnp.zeros((capacity,))
        self.next_states = jnp.zeros((capacity, state_dim))
        self.dones = jnp.zeros((capacity,))
        self.batch_size = batch_size

    @jax.jit
    def add_batch(self, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones):
        N     = batch_states.shape[0]
        idxs  = (self.ptr + jnp.arange(N)) % self.capacity
        
        new_states      = self.states.at[idxs].set(batch_states)
        new_actions     = self.actions.at[idxs].set(batch_actions)
        new_rewards     = self.rewards.at[idxs].set(batch_rewards)
        new_next_states = self.next_states.at[idxs].set(batch_next_states)
        new_dones       = self.dones.at[idxs].set(batch_dones)

        new_ptr  = (self.ptr  + N) % self.capacity
        new_size = jnp.minimum(self.size + N, self.capacity)

        children, aux = self.tree_flatten()
        new_children = (new_ptr, new_size, new_states, new_actions, new_rewards, new_next_states, new_dones)
        return self.tree_unflatten(aux, new_children)
    
    @jax.jit
    def add_batch_PPO(self, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones):
        N     = batch_states.shape[0]
        idxs  = (self.ptr + jnp.arange(N) * int(self.capacity // N))
        new_states      = self.states.at[idxs].set(batch_states)
        new_actions     = self.actions.at[idxs].set(batch_actions)
        new_rewards     = self.rewards.at[idxs].set(batch_rewards)
        new_next_states = self.next_states.at[idxs].set(batch_next_states)
        new_dones       = self.dones.at[idxs].set(batch_dones)

        new_ptr  = (self.ptr  + 1) % int(self.capacity // N)
        new_size = jnp.minimum(self.size + N, self.capacity)

        children, aux = self.tree_flatten()
        new_children = (new_ptr, new_size, new_states, new_actions, new_rewards, new_next_states, new_dones)
        return self.tree_unflatten(aux, new_children)

    @jax.jit
    def sample(self, rng: random.PRNGKey):

        idxs = jax.random.randint(rng, (self.batch_size,), 0, self.size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )
    
    def tree_flatten(self):
        children = (self.ptr, self.size, self.states, self.actions, self.rewards, self.next_states, self.dones)
        aux = (self.capacity, self.batch_size)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):

        obj = cls.__new__(cls)
        (obj.ptr, obj.size, obj.states, obj.actions, obj.rewards, obj.next_states, obj.dones) = children
        (obj.capacity, obj.batch_size) = aux
   
        return obj
