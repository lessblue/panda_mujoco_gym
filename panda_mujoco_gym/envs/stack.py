import os
import numpy as np
from panda_mujoco_gym.envs.panda_env import FrankaEnv
from gymnasium_robotics.utils import rotations

MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "../assets/", "stack.xml")


class FrankaStackEnv(FrankaEnv):
    def __init__(self, reward_type="sparse", **kwargs):
        self.object_height = 0.03
        self.obj1_name = "obj"
        self.obj2_name = "obj2"
        self.workspace_center = np.array([0.6, 0.0])
        
        super().__init__(
            model_path=MODEL_XML_PATH,
            reward_type=reward_type,
            n_substeps=25,
            block_gripper=False,
            distance_threshold=0.05,
            **kwargs,
        )

    def _get_obs(self) -> dict:
        ee_pos = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        ee_vel = self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy() * self.dt
        gripper_width = self.get_fingers_width().copy()
        obj1_pos = self._utils.get_site_xpos(self.model, self.data, "obj_site").copy()
        obj1_rot = rotations.mat2euler(self._utils.get_site_xmat(self.model, self.data, "obj_site")).copy()
        obj1_velp = self._utils.get_site_xvelp(self.model, self.data, "obj_site").copy() * self.dt
        obj1_velr = self._utils.get_site_xvelr(self.model, self.data, "obj_site").copy() * self.dt
        obj2_pos = self._utils.get_site_xpos(self.model, self.data, "obj2_site").copy()
        obj2_rot = rotations.mat2euler(self._utils.get_site_xmat(self.model, self.data, "obj2_site")).copy()
        obj2_velp = self._utils.get_site_xvelp(self.model, self.data, "obj2_site").copy() * self.dt
        obj2_velr = self._utils.get_site_xvelr(self.model, self.data, "obj2_site").copy() * self.dt
        observation = np.concatenate([ee_pos, ee_vel, gripper_width, obj1_pos, obj1_rot, obj1_velp, obj1_velr, obj2_pos, obj2_rot, obj2_velp, obj2_velr])
        return {"observation": observation.copy(), "achieved_goal": obj2_pos.copy(), "desired_goal": self.goal.copy()}

    def _reset_sim(self) -> bool:
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        self.set_joint_neutral()
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        base_pos = self.goal.copy() - np.array([0.0, 0.0, self.object_height])
        self._utils.set_joint_qpos(self.model, self.data, f"{self.obj1_name}_joint", np.concatenate([base_pos, [1.0, 0.0, 0.0, 0.0]]))

        while True:
            obj2_pos = self._sample_object_pos()
            if np.linalg.norm(obj2_pos - base_pos) > 0.1:
                break
        self._utils.set_joint_qpos(self.model, self.data, f"{self.obj2_name}_joint", np.concatenate([obj2_pos, [1.0, 0.0, 0.0, 0.0]]))
        
        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _sample_goal(self) -> np.ndarray:
        base_pos = self._utils.get_site_xpos(self.model, self.data, "obj_site").copy()
        goal = base_pos + np.array([0.0, 0.0, self.object_height])
        return goal.copy()
        
    def _sample_object_pos(self) -> np.ndarray:
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        pos = self.workspace_center.copy()
        pos += noise[:2] 
        pos = np.concatenate([pos, [self.initial_object_height]])
        return pos
