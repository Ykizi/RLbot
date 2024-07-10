import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution, CategoricalDistribution
from gymnasium.spaces import Box
import numpy as np

# 任务编码网络，用于生成任务的嵌入表示
class TaskEncodingNetwork(th.nn.Module):
    def __init__(self, num_tasks, embedding_dim):
        super(TaskEncodingNetwork, self).__init__()
        # 生成 num_tasks 个嵌入，每个嵌入的维度为 embedding_dim
        self.task_embedding = th.nn.Embedding(num_tasks, embedding_dim)

    def forward(self, task_id):
        # 根据任务 ID 返回相应的嵌入
        return self.task_embedding(task_id)

# 组合编码网络，用于结合观察和任务嵌入生成特征
class CompositionalEncodingNetwork(th.nn.Module):
    def __init__(self, input_dim, task_embedding_dim, hidden_dim):
        super(CompositionalEncodingNetwork, self).__init__()
        # 输入维度为观察空间维度加任务嵌入维度，隐藏层维度为 hidden_dim
        self.fc1 = th.nn.Linear(input_dim + task_embedding_dim, hidden_dim)
        self.fc2 = th.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, obs, task_embedding):
        # 将观察和任务嵌入连接起来，输入到全连接层中
        x = th.cat([obs, task_embedding], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# Actor 分布组合网络，用于生成动作分布
class ActorDistributionCompositionalNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, action_space, num_tasks, task_embedding_dim, hidden_dim, features_dim=256):
        super(ActorDistributionCompositionalNetwork, self).__init__(observation_space, features_dim)

        self._action_space = action_space

        # 输入维度为观察空间的形状
        input_dim = np.prod(observation_space.shape)
        # 任务编码网络
        self.task_encoding_net = TaskEncodingNetwork(num_tasks, task_embedding_dim)
        # 组合编码网络
        self.compositional_net = CompositionalEncodingNetwork(input_dim, task_embedding_dim, hidden_dim)

        # 根据动作空间选择适当的分布类型
        if isinstance(action_space, Box):
            self._projection_net = DiagGaussianDistribution(action_space.shape[0])
        else:
            self._projection_net = CategoricalDistribution(action_space.n)

    def forward(self, observations, task_id):
        # 获取任务嵌入并生成组合特征
        task_embedding = self.task_encoding_net(task_id)
        features = self.compositional_net(observations, task_embedding)
        return features

    def get_action_distribution(self, observations, task_id):
        # 根据组合特征生成动作分布
        features = self.forward(observations, task_id)
        return self._projection_net.proba_distribution(features)

# 组合评论家网络，用于估算 Q 值
class CompositionalCriticNetwork(th.nn.Module):
    def __init__(self, observation_space, action_space, num_tasks, task_embedding_dim, hidden_dim, features_dim=256):
        super(CompositionalCriticNetwork, self).__init__()
        self.features_dim = features_dim
        self._action_space = action_space

        # 输入维度为观察空间的形状
        input_dim = np.prod(observation_space.shape)
        # 动作空间维度
        action_dim = action_space.shape[0] if isinstance(action_space, Box) else action_space.n

        # 任务编码网络
        self.task_encoding_net = TaskEncodingNetwork(num_tasks, task_embedding_dim)
        # 组合编码网络
        self.compositional_net = CompositionalEncodingNetwork(input_dim, task_embedding_dim, hidden_dim)

        # Q 网络的全连接层
        self.q1_net = th.nn.Sequential(
            th.nn.Linear(hidden_dim + action_dim, features_dim),
            th.nn.ReLU(),
            th.nn.Linear(features_dim, 1)
        )
        self.q2_net = th.nn.Sequential(
            th.nn.Linear(hidden_dim + action_dim, features_dim),
            th.nn.ReLU(),
            th.nn.Linear(features_dim, 1)
        )

    def forward(self, observations, actions, task_id):
        # 获取任务嵌入并生成组合特征
        task_embedding = self.task_encoding_net(task_id)
        features = self.compositional_net(observations, task_embedding)
        # 将组合特征与动作连接起来作为 Q 网络的输入
        q1_input = th.cat([features, actions], dim=-1)
        q2_input = th.cat([features, actions], dim=-1)

        # 计算 Q 值
        q1 = self.q1_net(q1_input)
        q2 = self.q2_net(q2_input)

        return q1, q2
