import mujoco
from mujoco import mjx
from jax import lax
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
import jax.tree_util
from functools import partial
from Robot_Models.booster_t1.booster import get_obs_and_reward_walking

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Sensor:
    id: str     = field(metadata={'static': True})
    start: int     = field(metadata={'static': True})
    length: int     = field(metadata={'static': True})

@jax.tree_util.register_dataclass
@dataclass
class MODEL:
    body_mass: jax.Array 
    body_inertia: jax.Array 
    body_ipos: jax.Array 
    geom_friction: jax.Array 
    dof_frictionloss: jax.Array 
    dof_armature: jax.Array 

 
@jax.tree_util.register_dataclass
@dataclass
class ENVS:
    mjx_data: mjx.Data
    model: MODEL
    curr_action: jax.Array
    prev_action: jax.Array
    step_num: jax.Array
    stiffness: jax.Array
    damping: jax.Array
    force_applied: jax.Array
    goal_velocity: jax.Array
    key: jax.Array 

 
@jax.tree_util.register_pytree_node_class
class Sim:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mj_model = mujoco.MjModel.from_xml_path(cfg["PPO"]["xml_path"])
        self.mj_model.opt.iterations = 20
        self.timestep = self.mj_model.opt.timestep
        self.mj_model.opt.tolerance  = 1e-8

        self.mjx_model = mjx.put_model(self.mj_model)

        self.body_id = mjx.name2id(self.mjx_model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")

    @jax.jit
    def generate_mjx_model(self, key):
        mjx_model = self.mjx_model

        key, subkey = jax.random.split(key)
        body_mass = mjx_model.body_mass * (1. + jax.random.normal(subkey,  mjx_model.body_mass.shape) * self.cfg["STD"]["std_body_mass"])
        body_mass = jnp.clip(body_mass, a_min=0.0)

        key, subkey = jax.random.split(key)
        body_inertia = mjx_model.body_inertia * (1. + jax.random.normal(subkey,  mjx_model.body_inertia.shape) * self.cfg["STD"]["std_body_inertia"])
        body_inertia = jnp.clip(body_inertia, a_min=0)

        key, subkey = jax.random.split(key)
        delta = (jax.random.normal(subkey,  mjx_model.body_ipos.shape) * self.cfg["STD"]["std_body_ipos"]).clip(-.02, .02)
        body_ipos = mjx_model.body_ipos + delta
        
        key, subkey = jax.random.split(key)
        geom_friction = mjx_model.geom_friction * (1. + jax.random.normal(subkey,  mjx_model.geom_friction.shape) * self.cfg["STD"]["std_geom_friction"])
        geom_friction = jnp.clip(geom_friction, a_min=0.0, a_max=1.0)

        key, subkey = jax.random.split(key)
        dof_frictionloss = mjx_model.dof_frictionloss * (1. + jax.random.normal(subkey,  mjx_model.dof_frictionloss.shape) * self.cfg["STD"]["std_dof_frictionloss"])
        dof_frictionloss = jnp.clip(dof_frictionloss, a_min=0.0)

        key, subkey = jax.random.split(key)
        dof_armature = mjx_model.dof_armature * (1. + jax.random.normal(subkey,  mjx_model.dof_armature.shape) * self.cfg["STD"]["std_dof_armature"])
        dof_armature = jnp.clip(dof_armature, a_min=0.0)

        model = MODEL(body_mass, body_inertia, body_ipos, geom_friction, dof_frictionloss, dof_armature)

        return model

    @partial(jax.vmap, in_axes=(None, 0, 0), out_axes=(0))
    @jax.jit
    def step(self, envs: ENVS, action):
        mjx_model = self.mjx_model
        stiffness = envs.stiffness
        damping = envs.damping
        mjx_data = envs.mjx_data
        model = envs.model

        phyics_steps_per_model = int((1/self.cfg["PPO"]["model_freq"]) / self.timestep)
        step_num = envs.step_num + 1

        xfrc_applied_body = jnp.zeros(mjx_data.xfrc_applied[self.body_id].shape).at[3:5].set(envs.force_applied)
        xfrc_applied = jnp.zeros(mjx_data.xfrc_applied.shape).at[self.body_id].set(xfrc_applied_body)

        mjx_model = mjx_model.replace(body_mass = model.body_mass,
                                        body_inertia = model.body_inertia,
                                        body_ipos = model.body_ipos,
                                        geom_friction = model.geom_friction,
                                        dof_frictionloss = model.dof_frictionloss,
                                        dof_armature = model.dof_armature)
        
        def __step(context, _):
            d, m, action, xfrc_applied = context

            joint_pos = d.qpos[7:]
            joint_vel = d.qvel[6:]
            ctrl = stiffness * (action - joint_pos) - damping * (joint_vel)
            ctrl = ctrl.clip(-jnp.array(self.cfg["PPO"]["torque_limit"]), jnp.array(self.cfg["PPO"]["torque_limit"]))
            d = d.replace(ctrl = ctrl, xfrc_applied = xfrc_applied)
            
            d = mjx.step(m, d)  
            return (d, m, action, xfrc_applied), None
        
        (d, _, _, _), _ = jax.lax.scan(__step, (envs.mjx_data, mjx_model, action, xfrc_applied), None, length=phyics_steps_per_model)  

        force_applied = envs.force_applied * 0
        
        return ENVS(d, envs.model, action, envs.curr_action, step_num, envs.stiffness, envs.damping, force_applied, envs.goal_velocity, envs.key)
    
    # From mujoco playground
    @jax.jit
    def get_collision(self, d, geom1: int, geom2: int):
        contact = d._impl.contact
        mask = (jnp.array([geom1, geom2]) == contact.geom).all(axis=1)
        mask |= (jnp.array([geom2, geom1]) == contact.geom).all(axis=1)
        idx = jnp.where(mask, contact.dist, 1e4).argmin()
        dist = contact.dist[idx] * mask[idx]
        # normal = (dist < 0) * contact.frame[idx, 0, :3]
        return dist < 0
    
    @jax.jit
    def reset(self, key):
        mj_data = mujoco.MjData(self.mj_model)

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, int(self.cfg["PPO"]["batch_size"]))

        @jax.vmap
        def _reset(key):
            mjx_data = mjx.put_data(self.mj_model, mj_data)

            key, subkey = jax.random.split(key)
            qpos = jnp.array(self.cfg["PPO"]["init_pos"]) + jax.random.normal(subkey,  jnp.array(self.cfg["PPO"]["init_pos"]).shape) * self.cfg["STD"]["std_joint_pos"]
            key, subkey = jax.random.split(key)
            qvel = jnp.array(self.cfg["PPO"]["init_vel"]) + jax.random.normal(subkey,  jnp.array(self.cfg["PPO"]["init_vel"]).shape) * self.cfg["STD"]["std_joint_vel"]
            
            mjx_data = mjx_data.replace(qpos = qpos, qvel = qvel)

            key, subkey = jax.random.split(key)
            model = self.generate_mjx_model(subkey)

            key, subkey = jax.random.split(key)
            stiffness = jnp.array(self.cfg["PPO"]["stiffness"]) * (1 + jax.random.normal(subkey,  jnp.array(self.cfg["PPO"]["stiffness"]).shape) * self.cfg["STD"]["std_stiffness"])
            stiffness = jnp.clip(stiffness, a_min=0.0)

            key, subkey = jax.random.split(key)
            damping = jnp.array(self.cfg["PPO"]["damping"]) * (1 + jax.random.normal(subkey,  jnp.array(self.cfg["PPO"]["damping"]).shape) * self.cfg["STD"]["std_damping"])
            damping = jnp.clip(damping, a_min=0.0)
            
            force_applied = jnp.zeros(mjx_data.xfrc_applied[self.body_id][3:5].shape)

            step_num = 0

            curr_action = jnp.array(self.cfg["PPO"]["default_qpos"])
            prev_action = jnp.array(self.cfg["PPO"]["default_qpos"])

            key, subkey = jax.random.split(key)
            goal_velocity = jax.random.normal(subkey,  (3,)) * self.cfg["STD"]["std_goal_velocity"]

            return ENVS(mjx_data, model, curr_action, prev_action, step_num, stiffness, damping, force_applied, goal_velocity, key)

        return _reset(keys)
    
  
    @jax.jit
    def reset_partial(self, envs: ENVS, dones):

        @jax.vmap
        def _reset(env: ENVS, done):
            d = env.mjx_data
            m = env.model
            key = env.key

            mask = (done > 0)

            key, subkey = jax.random.split(key)
            qpos = jnp.array(self.cfg["PPO"]["init_pos"]) + jax.random.normal(subkey,  jnp.array(self.cfg["PPO"]["init_pos"]).shape) * self.cfg["STD"]["std_joint_pos"]
            key, subkey = jax.random.split(key)
            qvel = jnp.array(self.cfg["PPO"]["init_vel"]) + jax.random.normal(subkey,  jnp.array(self.cfg["PPO"]["init_vel"]).shape) * self.cfg["STD"]["std_joint_vel"]
            
            d = d.replace(
                qpos=jnp.where(mask, qpos, d.qpos),
                qvel=jnp.where(mask, qvel, d.qvel),
            )   

            key, subkey = jax.random.split(key)
            m = jax.lax.cond(
                mask,
                lambda _: self.generate_mjx_model(subkey),
                lambda _: m,
                operand=None,
            )

            key, subkey = jax.random.split(env.key)
            stiffness = jnp.where(mask, jnp.array(self.cfg["PPO"]["stiffness"]) * (1 + jax.random.normal(subkey,  jnp.array(self.cfg["PPO"]["stiffness"]).shape) * self.cfg["STD"]["std_stiffness"]), env.stiffness)
            stiffness = jnp.clip(stiffness, a_min=0.0)
           
            key, subkey = jax.random.split(key)
            damping = jnp.where(mask, jnp.array(self.cfg["PPO"]["damping"]) * (1 + jax.random.normal(subkey,  jnp.array(self.cfg["PPO"]["damping"]).shape) * self.cfg["STD"]["std_damping"]), env.damping)
            damping = jnp.clip(damping, a_min=0.0)

            key, subkey = jax.random.split(key)
            force_activate = jax.random.bernoulli(key, .03)
            key, subkey = jax.random.split(key)
            force_applied = jax.random.normal(subkey,  d.xfrc_applied[self.body_id][3:5].shape) * self.cfg["STD"]["std_force"]
            force_applied = jnp.where(force_applied > 0,
                    jnp.maximum(force_applied, self.cfg["STD"]["std_force"] / 2),
                    jnp.minimum(force_applied, -self.cfg["STD"]["std_force"] / 2))  * force_activate

            step_num = jnp.where(mask, 0, env.step_num)

            curr_action = jnp.where(mask, jnp.array(self.cfg["PPO"]["default_qpos"]), env.curr_action)
            prev_action = jnp.where(mask, jnp.array(self.cfg["PPO"]["default_qpos"]), env.prev_action)

            key, subkey = jax.random.split(key)
            goal_velocity = jnp.where(mask, jax.random.normal(subkey,  (3,)) * self.cfg["STD"]["std_goal_velocity"], env.goal_velocity)

            return ENVS(d, m, curr_action, prev_action, step_num, stiffness, damping, force_applied, goal_velocity, key)
        
        return _reset(envs, dones)
    
    @jax.jit
    def getObs_and_reward(self, envs: ENVS, key):

        keys = jax.random.split(key, int(self.cfg["PPO"]["batch_size"]))

        @partial(jax.vmap, in_axes=(0, 0), out_axes=(0))
        def _get_obs_and_reward(env: ENVS, key):
            obs, reward, done = get_obs_and_reward_walking(env, self, key)
            
            return obs, reward, done
        
        return _get_obs_and_reward(envs, keys)

    def tree_flatten(self):
        return (), (self.cfg, self.mj_model, self.mjx_model, self.timestep, self.body_id)

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, mj_model, mjx_model, timestep, body_id = aux
        obj = cls.__new__(cls)
        obj.cfg, obj.mj_model, obj.mjx_model, obj.timestep, obj.body_id = cfg, mj_model, mjx_model, timestep, body_id
        return obj
    
