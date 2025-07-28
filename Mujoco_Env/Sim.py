import mujoco
from mujoco import mjx
from jax import lax
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
import jax.tree_util
from functools import partial


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SimCfg:
    xml_path: str     = field(metadata={'static': True})
    batch: int     = field(metadata={'static': True})
    model_freq: int     = field(metadata={'static': True})
    
    init_pos: int     = field(metadata={'static': True})
    init_vel: int     = field(metadata={'static': True})

    std_gyro: int     = field(metadata={'static': True})
    std_acc: int     = field(metadata={'static': True})
    std_joint_pos: int     = field(metadata={'static': True})
    std_joint_vel: int     = field(metadata={'static': True})

    std_body_mass: int     = field(metadata={'static': True})
    std_body_inertia: int     = field(metadata={'static': True})
    std_body_ipos: int     = field(metadata={'static': True})
    std_geom_friction: int     = field(metadata={'static': True})
    std_dof_armature: int     = field(metadata={'static': True})
    std_dof_frictionloss: int     = field(metadata={'static': True})
    std_stiffness: int     = field(metadata={'static': True})
    std_damping: int     = field(metadata={'static': True})

    std_qpos: int     = field(metadata={'static': True})
    std_qvel: int     = field(metadata={'static': True})
    std_force: int     = field(metadata={'static': True})

    stiffness: jnp.array = field(metadata={'static': True})
    damping: jnp.array = field(metadata={'static': True})
    tau_limits: jnp.array = field(metadata={'static': True})

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Sensor:
    id: str     = field(metadata={'static': True})
    start: int     = field(metadata={'static': True})
    length: int     = field(metadata={'static': True})

@jax.tree_util.register_dataclass
@dataclass
class ENVS:
    mjx_data: mjx.Data
    mjx_model: mjx.Model
    step_num: jax.Array
    stiffness: jax.Array
    damping: jax.Array
    force_applied: jax.Array
    key: jax.Array 

 
@jax.tree_util.register_pytree_node_class
class Sim:
    def __init__(self, cfg: SimCfg):
        self.cfg = cfg
        self.mj_model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.mj_model.opt.iterations = 20
        self.timestep = self.mj_model.opt.timestep
        self.mj_model.opt.tolerance  = 1e-8

        self.mjx_model = mjx.put_model(self.mj_model)

        ang_vel_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
        self.ang_vel_sensor = Sensor(ang_vel_id, self.mjx_model.sensor_adr[ang_vel_id], self.mjx_model.sensor_dim[ang_vel_id])

        lin_accel_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "linear-acceleration")
        self.lin_accel_sensor = Sensor(lin_accel_id, self.mjx_model.sensor_adr[lin_accel_id], self.mjx_model.sensor_dim[lin_accel_id])

        self.right_foot_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")
        self.left_foot_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
        self.body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")

        self.right_foot_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom")
        self.left_foot_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom")
        self.ground_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    @jax.jit
    def generate_mjx_model(self, key):
        mjx_model = self.mjx_model

        key, subkey = jax.random.split(key)
        body_mass = mjx_model.body_mass * (1. + jax.random.normal(subkey,  mjx_model.body_mass.shape) * self.cfg.std_body_mass)
        body_mass = jnp.clip(body_mass, a_min=0.0)

        key, subkey = jax.random.split(key)
        body_inertia = mjx_model.body_inertia * (1. + jax.random.normal(subkey,  mjx_model.body_inertia.shape) * self.cfg.std_body_inertia)
        body_inertia = jnp.clip(body_inertia, a_min=0)

        key, subkey = jax.random.split(key)
        delta = (jax.random.normal(subkey,  mjx_model.body_ipos.shape) * self.cfg.std_body_ipos).clip(-.02, .02)
        body_ipos = mjx_model.body_ipos + delta
        
        key, subkey = jax.random.split(key)
        geom_friction = mjx_model.geom_friction * (1. + jax.random.normal(subkey,  mjx_model.geom_friction.shape) * self.cfg.std_geom_friction)
        geom_friction = jnp.clip(geom_friction, a_min=0.0, a_max=1.0)

        key, subkey = jax.random.split(key)
        dof_frictionloss = mjx_model.dof_frictionloss + (1. + jax.random.normal(subkey,  mjx_model.dof_frictionloss.shape) * self.cfg.std_dof_frictionloss)
        dof_frictionloss = jnp.clip(dof_frictionloss, a_min=0.0)

        key, subkey = jax.random.split(key)
        dof_armature = mjx_model.dof_armature + (1. + jax.random.normal(subkey,  mjx_model.dof_armature.shape) * self.cfg.std_dof_armature)
        dof_armature = jnp.clip(dof_armature, a_min=0.0)

        mjx_model = mjx_model.replace(body_mass = body_mass,
                                        body_inertia = body_inertia,
                                        body_ipos = body_ipos,
                                        geom_friction = geom_friction,
                                        dof_frictionloss = dof_frictionloss,
                                        dof_armature = dof_armature)
        
        return mjx_model

    @partial(jax.vmap, in_axes=(None, 0, 0), out_axes=(0))
    @jax.jit
    def step(self, envs: ENVS, action):
        stiffness = envs.stiffness
        damping = envs.damping
        mjx_data = envs.mjx_data

        phyics_steps_per_model = int((1/self.cfg.model_freq) / self.timestep)
        step_num = envs.step_num + 1

        xfrc_applied_body = jnp.zeros(mjx_data.xfrc_applied[self.body_id].shape).at[3:].set(envs.force_applied)
        xfrc_applied = jnp.zeros(mjx_data.xfrc_applied.shape).at[self.body_id].set(xfrc_applied_body)
        
        def __step(context, _):
            d, m, action, xfrc_applied = context

            joint_pos = d.qpos[7:]
            joint_vel = d.qvel[6:]
            ctrl = stiffness * (action - joint_pos) - damping * (joint_vel)
            ctrl = ctrl.clip(-self.cfg.tau_limits, self.cfg.tau_limits)
            d = d.replace(ctrl = ctrl, xfrc_applied = xfrc_applied)
            
            d = mjx.step(m, d)  
            return (d, m, action, xfrc_applied), None
        
        (d, _, _, _), _ = jax.lax.scan(__step, (envs.mjx_data, envs.mjx_model, action, xfrc_applied), None, length=phyics_steps_per_model)  
        
        return ENVS(d, envs.mjx_model, step_num, envs.stiffness, envs.damping, envs.force_applied, envs.key)
    
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
        keys = jax.random.split(subkey, int(self.cfg.batch))

        @jax.vmap
        def _reset(key):
            mjx_data = mjx.put_data(self.mj_model, mj_data)

            key, subkey = jax.random.split(key)
            qpos = self.cfg.init_pos + jax.random.normal(subkey,  self.cfg.init_pos.shape) * self.cfg.std_qpos
            key, subkey = jax.random.split(key)
            qvel = self.cfg.init_vel + jax.random.normal(subkey,  self.cfg.init_vel.shape) * self.cfg.std_qvel
            
            mjx_data = mjx_data.replace(qpos = qpos, qvel = qvel)

            mjx_model = self.generate_mjx_model(key)

            key, subkey = jax.random.split(key)
            stiffness = self.cfg.stiffness * (1 + jax.random.normal(subkey,  self.cfg.stiffness.shape) * self.cfg.std_stiffness)
            stiffness = jnp.clip(stiffness, a_min=0.0)

            key, subkey = jax.random.split(key)
            damping = self.cfg.damping * (1 + jax.random.normal(subkey,  self.cfg.damping.shape) * self.cfg.std_damping)
            damping = jnp.clip(damping, a_min=0.0)
            
            key, subkey = jax.random.split(key)
            force_applied = jax.random.normal(subkey,  mjx_data.xfrc_applied[self.body_id][3:].shape) * self.cfg.std_force

            step_num = 0

            return ENVS(mjx_data, mjx_model, step_num, stiffness, damping, force_applied, subkey)

        return _reset(keys)
    
  
    @jax.jit
    def reset_partial(self, envs: ENVS, dones):

        @jax.vmap
        def _reset(env: ENVS, done):
            d = env.mjx_data
            m = env.mjx_model

            mask = (done > 0)

            key, subkey = jax.random.split(env.key)
            qpos = self.cfg.init_pos + jax.random.normal(subkey,  self.cfg.init_pos.shape) * self.cfg.std_qpos
            key, subkey = jax.random.split(key)
            qvel = self.cfg.init_vel + jax.random.normal(subkey,  self.cfg.init_vel.shape) * self.cfg.std_qvel

            d = d.replace(
                qpos=jnp.where(mask, qpos, d.qpos),
                qvel=jnp.where(mask, qvel, d.qvel),
            )   

            m = jax.lax.cond(
                mask,
                lambda _: self.generate_mjx_model(key),
                lambda _: m,
                operand=None,
            )

            key, subkey = jax.random.split(env.key)
            stiffness = jnp.where(mask, self.cfg.stiffness * (1 + jax.random.normal(subkey,  self.cfg.stiffness.shape) * self.cfg.std_stiffness), env.stiffness)
            stiffness = jnp.clip(stiffness, a_min=0.0)
           
            key, subkey = jax.random.split(key)
            damping = jnp.where(mask, self.cfg.damping * (1 + jax.random.normal(subkey,  self.cfg.damping.shape) * self.cfg.std_damping), env.damping)
            damping = jnp.clip(damping, a_min=0.0)

            key, subkey = jax.random.split(key)
            force_applied = jnp.where(mask, jax.random.normal(subkey,  d.xfrc_applied[self.body_id][3:].shape) * self.cfg.std_force, env.force_applied)

            step_num = jnp.where(done > 0, 0, env.step_num)

            return ENVS(d, m, step_num, stiffness, damping, force_applied, key)
        
        return _reset(envs, dones)
    
    @jax.jit
    def getObs(self, envs: ENVS, key):

        keys = jax.random.split(key, int(self.cfg.batch))

        @partial(jax.vmap, in_axes=(0, 0), out_axes=(0))
        def _get_obs(d, key):
            body_pos = d.qpos[:3]
            quat = d.qpos[3:7]
            body_vel = d.qvel[:3]
            joint_pos = d.qpos[7:]
            joint_vel = d.qvel[6:]

            base_ang_vel = d.sensordata[self.ang_vel_sensor.start:self.ang_vel_sensor.start + self.ang_vel_sensor.length]
            base_lin_accel = d.sensordata[self.lin_accel_sensor.start:self.lin_accel_sensor.start + self.lin_accel_sensor.length]

            right_foot_pos = d.xpos[self.right_foot_body_id]
            right_foot_vel = d.cvel[self.right_foot_body_id][3:]
            right_foot_ang_vel = d.cvel[self.right_foot_body_id][:3]
            right_foot_force = d._impl.cfrc_ext[self.right_foot_body_id][3:]
            right_foot_moment = d._impl.cfrc_ext[self.right_foot_body_id][:3]
            right_foot_ground_contact = self.get_collision(d, self.right_foot_geom_id, self.ground_geom_id)

            left_foot_pos = d.xpos[self.left_foot_body_id]
            left_foot_vel = d.cvel[self.left_foot_body_id][3:]
            left_foot_ang_vel = d.cvel[self.left_foot_body_id][:3]
            left_foot_force = d._impl.cfrc_ext[self.left_foot_body_id][3:]
            left_foot_ang_moment = d._impl.cfrc_ext[self.left_foot_body_id][:3]
            left_foot_ground_contact = self.get_collision(d, self.left_foot_geom_id, self.ground_geom_id)

            prev_ctrl = d.ctrl[:]

            key, subkey = jax.random.split(key)
            noise_joint_pos = self.cfg.std_joint_pos * jax.random.normal(subkey, joint_pos.shape)
            joint_pos = joint_pos + noise_joint_pos

            key, subkey = jax.random.split(key)
            noise_joint_vel = self.cfg.std_joint_vel * jax.random.normal(subkey, joint_vel.shape)
            joint_vel = joint_vel + noise_joint_vel

            key, subkey = jax.random.split(key)
            noise_ang_vel = self.cfg.std_gyro * jax.random.normal(subkey, base_ang_vel.shape)
            base_ang_vel = base_ang_vel + noise_ang_vel

            key, subkey = jax.random.split(key)
            noise_lin_accel = self.cfg.std_acc * jax.random.normal(subkey, base_lin_accel.shape)
            base_lin_accel = base_lin_accel + noise_lin_accel

            obs = jnp.concatenate([
                prev_ctrl, joint_pos, joint_vel, base_ang_vel, base_lin_accel, quat, 
                body_pos, body_vel,
                right_foot_pos, left_foot_pos,
                right_foot_vel, left_foot_vel,
                right_foot_ang_vel, left_foot_ang_vel,
                right_foot_force, left_foot_force,
                right_foot_moment, left_foot_ang_moment,
                jnp.array([right_foot_ground_contact, left_foot_ground_contact])
            ], axis=-1)
            
            return obs
        
        return _get_obs(envs.mjx_data, keys)


    def tree_flatten(self):
        return (), (self.cfg, self.mj_model, self.mjx_model, self.timestep, self.ang_vel_sensor, self.lin_accel_sensor, self.right_foot_body_id, self.left_foot_body_id, self.body_id, self.right_foot_geom_id, self.left_foot_geom_id, self.ground_geom_id)

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, mj_model, mjx_model, timestep, ang_vel_sensor, lin_accel_sensor, right_foot_body_id, left_foot_body_id, body_id, right_foot_geom_id, left_foot_geom_id, ground_geom_id = aux
        obj = cls.__new__(cls)
        obj.cfg, obj.mj_model, obj.mjx_model, obj.timestep, obj.ang_vel_sensor, obj.lin_accel_sensor, obj.right_foot_body_id, obj.left_foot_body_id, obj.body_id, obj.right_foot_geom_id, obj.left_foot_geom_id, obj.ground_geom_id = cfg, mj_model, mjx_model, timestep, ang_vel_sensor, lin_accel_sensor, right_foot_body_id, left_foot_body_id, body_id, right_foot_geom_id, left_foot_geom_id, ground_geom_id
        return obj
    
