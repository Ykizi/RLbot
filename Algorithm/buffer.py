from stable_baselines3.common.buffers import ReplayBuffer
import torch as th
import numpy as np

class CustomReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device="cpu", n_envs=1, optimize_memory_usage=False):
        super().__init__(buffer_size, observation_space, action_space, device=device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage)
        self.task_ids = np.zeros((buffer_size, n_envs), dtype=np.int32)  # 添加 task_ids 属性

    def add(self, obs, next_obs, action, reward, done, infos, task_id):  # 添加 infos 参数
        # 添加此行进行调试
        if not isinstance(infos, list) or not all(isinstance(info, dict) for info in infos):
            raise ValueError(f"Expected infos to be a list of dictionaries, but got: {infos}")
        # 调用父类的 add 方法存储其他数据
        super().add(obs, next_obs, action, reward, done, infos)
        # 存储 task_id
        self.task_ids[self.pos] = task_id

    def sample(self, batch_size):
        # 调用父类的 sample 方法获取采样的数据
        replay_data = super().sample(batch_size)
        max_pos = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, max_pos, size=batch_size)

        # 将 task_id 添加到采样数据中
        replay_data.task_ids = th.tensor(self.task_ids[batch_inds]).to(self.device)
        return replay_data

