import numpy as np
from robopal.demos.manipulation_tasks.robot_manipulate_dense import ManipulateDenseEnv
import robopal.commons.transform as T
from robopal.robots.ur5e import UR5eGrasp


class GraspHandlingEnv(ManipulateDenseEnv):

    def __init__(self,
                 robot=UR5eGrasp,
                 render_mode='human',
                 control_freq=20,
                 enable_camera_viewer=False,
                 controller='CARTIK',
                 ):
        super().__init__(
            robot=robot,
            render_mode=render_mode,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            controller=controller,
        )
        self.name = 'ConveyorHandling-v1'
        self.obs_dim = (22,)  # 更新观测空间维度以包含 One-Hot 编码
        self.goal_dim = (3,)
        self.action_dim = (4,)

        self.max_action = 1.0
        self.min_action = -1.0

        self.max_episode_steps = 500

        self.pos_max_bound = np.array([0.7, 0.6, 0.3])
        self.pos_min_bound = np.array([-0.4, -0.7, -0.13])
        self.grip_max_bound = 0.95
        self.grip_min_bound = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.get_body_pos('green_block')[2] > 0.5:
            terminated = True
        if self.get_body_pos('green_block')[2] < 0.1:
            terminated = True
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        """
        obs = np.zeros(self.obs_dim)

        obs[0:3] = (  # gripper position in global coordinates
            end_pos := self.get_site_pos('0_grip_site')
        )
        obs[3:6] = (  # block position in global coordinates
            object_pos := self.get_body_pos('green_block')
        )
        obs[6:9] = (  # Relative block position with respect to gripper position in globla coordinates.
            end_pos - object_pos
        )
        obs[9:12] = (  # block rotation
            T.mat_2_euler(self.get_body_rotm('green_block'))
        )
        obs[12:15] = (  # gripper linear velocity
            self.get_site_xvelp('0_grip_site') * self.dt
        )
        obs[15:18] = (  # block linear velocity
            self.get_body_xvelp('green_block') * self.dt
        )
        obs[18:21] = (  # block angular velocity
            self.get_body_xvelr('green_block') * self.dt
        )
        obs[21] = self.mj_data.joint('0_robotiq_2f_85_right_driver_joint').qpos[0]

        return obs.copy()

    def _get_info(self) -> dict:
        return {'is_success': self._is_success(self.get_body_pos('green_block'), self.get_body_pos('carton'), th=0.02)}

    def reset_object(self):
        random_x_pos = np.random.uniform(1.6, 1.75)
        self.set_object_pose('green_block:joint', np.array([random_x_pos, 0.3, 0.45, 1.0, 0.0, 0.0, 0.0]))

    def compute_rewards(self, info: dict = None, **kwargs):
        cube2gripper = self.goal_distance(self.get_body_pos('green_block'), self.get_site_pos('0_grip_site'))

        if cube2gripper <= 0.02:  # gripper has catched the cube
            reward = 50
        else:  # cube is near gripper
            reward = 0 - 10 * cube2gripper

        reward += 200 * (self.get_body_pos('green_block')[2] - 0.45)

        return reward



if __name__ == "__main__":
    env = GraspHandlingEnv()
    env.reset()
    obs = env.reset()
    print(obs)  # 检查重置后环境返回的观察值的形状

    for t in range(int(1e5)):
        action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
        s_, r, terminated, truncated, info = env.step(action)
        if terminated:
            env.reset()
    env.close()

    # # 检查环境的观测空间
    # env = make_env(env_id)
    # obs = env.reset()
    # print(f"Observation shape: {obs.shape}")
    #
    # # 确认 obs_dim 与观测空间匹配
    # self.obs_dim = env.observation_space.shape
    #
    # # 确认你的 CustomSAC 传递了正确的参数
    # model = CustomSAC(
    #     "MlpPolicy",
    #     env,
    #     num_tasks=3,
    #     task_embedding_dim=10,
    #     hidden_dim=256
    # )
    #
    # # 调试任务ID的处理
    # print(f"BaseTask ID: {replay_data.task_ids}")
