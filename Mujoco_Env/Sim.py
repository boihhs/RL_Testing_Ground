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

    stiffness: jnp.array = field(metadata={'static': True})
    damping: jnp.array = field(metadata={'static': True})
    tau_limits: jnp.array = field(metadata={'static': True})

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Sensor:
    id: str     = field(metadata={'static': True})
    start: int     = field(metadata={'static': True})
    length: int     = field(metadata={'static': True})
 

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

        self.right_foot_pos_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")
        self.left_foot_pos_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
        
        

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=(0, 0))
    def step(self, d, action, step_num):
        phyics_steps_per_model = int((1/self.cfg.model_freq) / self.timestep)

        step_num = step_num + 1
        
        def _step(context, _):
            d, action = context

            joint_pos = d.qpos[7:]
            joint_vel = d.qvel[6:]
            ctrl = self.cfg.stiffness * (action - joint_pos) - self.cfg.damping * (joint_vel)
            ctrl = ctrl.clip(-self.cfg.tau_limits, self.cfg.tau_limits)
            d = d.replace(ctrl = ctrl)
            
            d = mjx.step(self.mjx_model, d)  
            return (d, action), None
        
        (d, _), _ = jax.lax.scan(_step, (d, action), None, length=phyics_steps_per_model)
        
        
        return d, step_num
    
    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=(0))
    def get_collision(self, d, geom1: int, geom2: int):
        contact = d._impl.contact
        mask = (jnp.array([geom1, geom2]) == contact.geom).all(axis=1)
        mask |= (jnp.array([geom2, geom1]) == contact.geom).all(axis=1)
        idx = jnp.where(mask, contact.dist, 1e4).argmin()
        dist = contact.dist[idx] * mask[idx]
        normal = (dist < 0) * contact.frame[idx, 0, :3]
        return dist < 0
    
    @jax.jit
    def reset(self):
        mj_data = mujoco.MjData(self.mj_model)

        @partial(jax.vmap, in_axes=(0), out_axes=(0))
        def _reset(_):
            mjx_data = mjx.put_data(self.mj_model, mj_data)
            mjx_data = mjx_data.replace(qpos = self.cfg.init_pos, qvel = self.cfg.init_vel)
            return mjx_data
    
        seq = jnp.arange(self.cfg.batch)
        return _reset(seq), jnp.zeros((self.cfg.batch,), dtype=jnp.int32)
    
  
    @jax.jit
    def reset_partial(self, d , dones, step_nums):
        @partial(jax.vmap, in_axes=(0, 0), out_axes=(0))
        def _reset(d, done):
            mask = (done > 0)
            d = d.replace(qpos = self.cfg.init_pos * mask + d.qpos * (1 - mask), qvel = self.cfg.init_vel * mask + d.qvel * (1 - mask))
            return d
    
        return _reset(d, dones), jnp.where(dones > 0, 0, step_nums)
    
    @jax.jit
    def getObs(self, d, key):
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
            right_foot_pos = d.xpos[self.right_foot_pos_id]
            left_foot_pos = d.xpos[self.left_foot_pos_id]

            prev_ctrl = d.ctrl[:]

            key, subkey = jax.random.split(key)
            noise_joint_pos = self.cfg.std_gyro * jax.random.normal(subkey, joint_pos.shape)
            joint_pos = joint_pos + noise_joint_pos

            key, subkey = jax.random.split(key)
            noise_joint_vel = self.cfg.std_acc * jax.random.normal(subkey, joint_vel.shape)
            joint_vel = joint_vel + noise_joint_vel

            key, subkey = jax.random.split(key)
            noise_ang_vel = self.cfg.std_gyro * jax.random.normal(subkey, base_ang_vel.shape)
            base_ang_vel = base_ang_vel + noise_ang_vel

            key, subkey = jax.random.split(key)
            noise_lin_accel = self.cfg.std_acc * jax.random.normal(subkey, base_lin_accel.shape)
            base_lin_accel = base_lin_accel + noise_lin_accel
            
            return jnp.concatenate([prev_ctrl, joint_pos, joint_vel, base_ang_vel, base_lin_accel, quat, body_pos, body_vel, right_foot_pos, left_foot_pos], axis=-1)
        
        return _get_obs(d, keys)


    def tree_flatten(self):
        return (), (self.cfg, self.mj_model, self.mjx_model, self.timestep, self.ang_vel_sensor, self.lin_accel_sensor, self.right_foot_pos_id, self.left_foot_pos_id)

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, mj_model, mjx_model, timestep, ang_vel_sensor, lin_accel_sensor, right_foot_pos_id, left_foot_pos_id = aux
        obj = cls.__new__(cls)
        obj.cfg, obj.mj_model, obj.mjx_model, obj.timestep, obj.ang_vel_sensor, obj.lin_accel_sensor, obj.right_foot_pos_id, obj.left_foot_pos_id = cfg, mj_model, mjx_model, timestep, ang_vel_sensor, lin_accel_sensor, right_foot_pos_id, left_foot_pos_id
        return obj
    
