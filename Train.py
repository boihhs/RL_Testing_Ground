
import mujoco
from mujoco import mjx
from jax import lax
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from jax.tree_util import register_pytree_node_class
import jax.tree_util
from RL_Algos.PPO import PPO

cfg_file = "/home/leo-benaharon/Desktop/HumanPose/RL_Testing_Ground/RL_Algos/PPO.yaml"
robot_data_path = "/home/leo-benaharon/Desktop/HumanPose/GMR/path_to_save_robot_data.pkl"


csac = PPO(cfg_file)
csac.run()
print(csac.buffer.size.item())

