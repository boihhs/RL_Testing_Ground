

import time, re, threading
from pathlib import Path

import mujoco
from mujoco import viewer
import yaml
import numpy as np
import jax.numpy as jnp


import pickle

# file_path = "/home/leo-benaharon/Desktop/HumanPose/GMR/path_to_save_robot_data.pkl"
class getGroundTurthPositions:
    def __init__(self,robot_data_path, cfg_file, DT_CURRENT = -1):
          
        # Load the pickle file
        robot_data_path = "/home/leo-benaharon/Desktop/HumanPose/GMR/path_to_save_robot_data.pkl"

        with open(robot_data_path, "rb") as f:
            self.robot_data = pickle.load(f)


        cfg_file = "/home/leo-benaharon/Desktop/HumanPose/RL_Testing_Ground/RL_Algos/PPO.yaml"
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)


        self.DT_TARGET = 1.0 / self.cfg["PPO"]["model_freq"]
        if DT_CURRENT == -1:
            self.DT_CURRENT = 1.0 / self.robot_data["fps"]
        else: 
            self.DT_CURRENT = DT_CURRENT

        self.length = self.robot_data["root_pos"].shape[0]
        self.time_action = (self.length - 1) * self.DT_CURRENT
        self.target_length = int(round(self.time_action / self.DT_TARGET)) + 1
        self.addon_length = int(.5 / self.DT_TARGET)

        self.root_pos = np.zeros((self.target_length + self.addon_length, 3))
        self.root_rot = np.zeros((self.target_length + self.addon_length, 4))
        self.dof_pos = np.zeros((self.target_length + self.addon_length, self.robot_data["dof_pos"].shape[1]))

        self.interpliate()

        self.root_vel = self.central_diff(self.root_pos, self.DT_TARGET)
        self.root_ang_vel = self.angular_vel_from_quats(self.root_rot, self.DT_TARGET)
        self.dof_vel = self.central_diff(self.dof_pos, self.DT_TARGET)

        xml_path = self.cfg["PPO"]["xml_path"]
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        num_bodys = self.mj_data.xpos.shape[0] - 1
        self.body_positions = np.zeros((self.target_length + self.addon_length, num_bodys, 3))
        self.body_rotations_quat = np.zeros((self.target_length + self.addon_length, num_bodys, 4))
        self.body_rotations_m = np.zeros((self.target_length + self.addon_length, num_bodys, 9))
        self.body_velcitys = np.zeros((self.target_length + self.addon_length, num_bodys, 3))
        self.body_ang_velitys = np.zeros((self.target_length + self.addon_length, num_bodys, 3))


    def central_diff(self, x, dt):
        # x: (T, D)
        v = np.empty_like(x)
        v[1:-1] = (x[2:] - x[:-2]) / (2*dt)
        v[0]    = (x[1]  - x[0])   / dt
        v[-1]   = (x[-1] - x[-2])  / dt
        return v

    def quat_log_rel(self, qb, qa):
        """Log of relative rotation qb*qa^{-1}, quats are [w,x,y,z]."""
        # quaternion conjugate (inverse for unit quats)
        qa_conj = np.array([qa[0], -qa[1], -qa[2], -qa[3]])
        # Hamilton product
        w = qb[0]*qa_conj[0] - qb[1]*qa_conj[1] - qb[2]*qa_conj[2] - qb[3]*qa_conj[3]
        x = qb[0]*qa_conj[1] + qb[1]*qa_conj[0] + qb[2]*qa_conj[3] - qb[3]*qa_conj[2]
        y = qb[0]*qa_conj[2] - qb[1]*qa_conj[3] + qb[2]*qa_conj[0] + qb[3]*qa_conj[1]
        z = qb[0]*qa_conj[3] + qb[1]*qa_conj[2] - qb[2]*qa_conj[1] + qb[3]*qa_conj[0]
        q_rel = np.array([w, x, y, z])

        # convert to axis-angle
        theta = 2 * np.arccos(np.clip(q_rel[0], -1.0, 1.0))
        if theta < 1e-8:
            return np.zeros(3)
        axis = q_rel[1:4] / np.linalg.norm(q_rel[1:4])
        return axis * theta


    def angular_vel_from_quats(self, q, dt):
        """Compute angular velocity from quaternion sequence, [w,x,y,z]."""
        q = q.copy()
        for k in range(1, len(q)):
            if np.dot(q[k], q[k-1]) < 0:  # continuity fix
                q[k] = -q[k]

        w = np.empty((len(q), 3))
        for k in range(1, len(q)-1):
            w[k] = self.quat_log_rel(q[k+1], q[k-1]) / (2*dt)
        w[0]  = self.quat_log_rel(q[1], q[0]) / dt
        w[-1] = self.quat_log_rel(q[-1], q[-2]) / dt
        return w


    def nlerp(self, p1, p2, a):
        return p1 * (1 - a) + p2 * a

    def slerp(self, q1, q2, a, eps=1e-8):
        """Spherical linear interpolation, quats are [w,x,y,z]."""
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)

        d = float(np.dot(q1, q2))
        if d < 0.0:
            q2 = -q2
            d = -d
        d = np.clip(d, -1.0, 1.0)

        theta = np.arccos(d)
        if theta < eps:
            return ((1.0 - a) * q1 + a * q2) / np.linalg.norm(q1)

        sin_theta = np.sin(theta)
        s0 = np.sin((1.0 - a) * theta) / sin_theta
        s1 = np.sin(a * theta) / sin_theta
        return s0 * q1 + s1 * q2

    
    def interpliate(self):
        t_in  = np.arange(self.length) * (self.DT_CURRENT)
        t_out = np.arange(self.target_length) * self.DT_TARGET

        u = np.clip(t_out / (t_in[-1] + 1e-12), 0.0, 1.0) * (self.length - 1)
        i0 = np.floor(u).astype(int)
        i1 = np.clip(i0 + 1, 0, self.length - 1)
        a  = (u - i0).reshape(-1, 1)  # (T,1)

        q_in = self.robot_data["root_rot"].copy()
        order = [3, 0, 1, 2]
        q_in = q_in[:, order]
        for k in range(1, len(q_in)):
            if np.dot(q_in[k], q_in[k-1]) < 0:
                q_in[k] = -q_in[k]

        self.root_pos[:self.target_length] = (1 - a) * self.robot_data["root_pos"][i0] + a * self.robot_data["root_pos"][i1]
        self.dof_pos[:self.target_length]  = (1 - a) * self.robot_data["dof_pos"][i0]  + a * self.robot_data["dof_pos"][i1]

        out_q = np.zeros((self.target_length, 4))
        for k in range(self.target_length):
            q0, q1, ak = q_in[i0[k]], q_in[i1[k]], float(u[k] - i0[k])
            if np.dot(q1, q0) < 0:
                q1 = -q1
            out_q[k] = self.slerp(q0, q1, ak)
        self.root_rot[:self.target_length] = out_q

        last = self.length - 1
        q_last, q_first = q_in[last].copy(), q_in[0].copy()
        if np.dot(q_last, q_first) < 0: 
            q_first = -q_first
        for j in range(self.addon_length):
            a = (j + 1) / (self.addon_length + 1)
            # self.root_pos[self.target_length + j] = (1 - a) * self.robot_data["root_pos"][last] + a * self.robot_data["root_pos"][0]
            self.root_pos[self.target_length + j] = self.robot_data["root_pos"][last]
            self.dof_pos[self.target_length + j]  = (1 - a) * self.robot_data["dof_pos"][last]  + a * self.robot_data["dof_pos"][0]
            self.root_rot[self.target_length + j] = self.slerp(q_last, q_first, a)
            # self.root_rot[self.target_length + j] = q_last


    def run(self):
        offset = np.array([0, 0, .0])
        self.root_pos = self.root_pos + offset[None, :]
        with viewer.launch_passive(self.mj_model, self.mj_data) as v:
            for i in range(self.target_length + self.addon_length):

                

                self.mj_data.qpos[:3] = self.root_pos[i]
                
                self.mj_data.qpos[3:7] = self.root_rot[i]
                self.mj_data.qpos[7:] = self.dof_pos[i]

                self.mj_data.qvel[:3]  = self.root_vel[i]
                self.mj_data.qvel[3:6] = self.root_ang_vel[i]
                self.mj_data.qvel[6:] = self.dof_vel[i]

                mujoco.mj_forward(self.mj_model, self.mj_data)


                self.body_positions[i] = self.mj_data.xpos[1:]
                self.body_velcitys[i] = self.mj_data.cvel[1:, 3:]
                self.body_rotations_quat[i] = self.mj_data.xquat[1:]
                self.body_rotations_m[i] = self.mj_data.xmat[1:]
                self.body_ang_velitys[i] = self.mj_data.cvel[1:, :3]
                time.sleep(self.DT_TARGET)
                if not v.is_running():
                    break
                v.sync()

        self.root_pos = jnp.asarray(self.root_pos)
        self.root_rot = jnp.asarray(self.root_rot)
        self.dof_pos = jnp.asarray(self.dof_pos)

        self.root_vel = jnp.asarray(self.root_vel)
        self.root_ang_vel = jnp.asarray(self.root_ang_vel)
        self.dof_vel = jnp.asarray(self.dof_vel)

        self.body_positions = jnp.asarray(self.body_positions)
        self.body_rotations_quat = jnp.asarray(self.body_rotations_quat)
        self.body_rotations_m = jnp.asarray(self.body_rotations_m)
        self.body_velcitys = jnp.asarray(self.body_velcitys)
        self.body_ang_velitys = jnp.asarray(self.body_ang_velitys)

        print("Got things for reward 🎲")

        

        