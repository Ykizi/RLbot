from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import get_parameters_by_name
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from Algorithm.network import ActorDistributionCompositionalNetwork, CompositionalCriticNetwork
from Algorithm.buffer import CustomReplayBuffer

# 定义一个泛型类型SelfSAC，用于类型检查
SelfSAC = TypeVar("SelfSAC", bound="SAC")

# 定义一个自定义的SAC算法类
class CustomSAC(OffPolicyAlgorithm):
    # 定义策略别名，将策略名称映射到具体的策略类
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # 声明一些类属性
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    # 初始化函数
    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = CustomReplayBuffer,  # 使用自定义的 ReplayBuffer
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        num_tasks: int = 4,  # 任务数量参数
        task_embedding_dim: int = 10,  # 任务编码维度参数
        hidden_dim: int = 256  # 隐藏层维度参数
    ):
        # 设置自定义的任务相关参数
        self.num_tasks = num_tasks
        self.task_embedding_dim = task_embedding_dim
        self.hidden_dim = hidden_dim

        # 调用父类的初始化方法，传递大部分参数
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        # 设置目标熵和熵系数相关的变量
        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None

        # 如果初始化标志为True，则调用模型的设置方法
        if _init_setup_model:
            self._setup_model()

    # 设置模型的方法
    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # 获取critic和critic_target中批量归一化层的参数
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

        # 自动设置目标熵值，如果指定为"auto"
        if self.target_entropy == "auto":
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            self.target_entropy = float(self.target_entropy)

        # 自动调整熵系数
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"
            # 创建一个可训练的张量log_ent_coef
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            # 创建一个Adam优化器来优化log_ent_coef
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # 如果指定了固定的熵系数，则将其转换为张量
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    # 创建别名的方法
    def _create_aliases(self) -> None:
        # 默认的特征维度
        features_dim = 256

        # 初始化Actor网络
        self.actor = ActorDistributionCompositionalNetwork(
            self.observation_space,
            self.action_space,
            self.num_tasks,
            self.task_embedding_dim,
            self.hidden_dim,
            features_dim=features_dim
        )
        # 初始化Critic网络
        self.critic = CompositionalCriticNetwork(
            self.observation_space,
            self.action_space,
            self.num_tasks,
            self.task_embedding_dim,
            self.hidden_dim,
            features_dim=features_dim
        )
        # 初始化目标Critic网络
        self.critic_target = CompositionalCriticNetwork(
            self.observation_space,
            self.action_space,
            self.num_tasks,
            self.task_embedding_dim,
            self.hidden_dim,
            features_dim=features_dim
        )

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, done, infos):
        """
        Store transition in the replay buffer.
        :param replay_buffer: Replay buffer object
        :param buffer_action: Action to store
        :param new_obs: New observation
        :param reward: Reward for the transition
        :param done: Is the episode done?
        :param infos: Extra information about the transition
        """
        print(f"infos: {infos}")  # 添加打印语句检查 infos 的内容

        # 遍历 infos 列表，获取每个 info 字典中的 task_id
        for idx, info in enumerate(infos):
            task_id = info.get('task_id', 0)  # 根据你的环境，修改获取 task_id 的方式

            # 将 transition 添加到 replay buffer 中
            replay_buffer.add(self._last_obs[idx], new_obs[idx], buffer_action[idx], reward[idx], done[idx], info,
                              task_id)  # 添加 task_id 参数

        # 更新当前的观测值
        self._last_obs = new_obs

    # 训练方法
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # 设置策略为训练模式
        self.policy.set_training_mode(True)
        # 获取需要优化的优化器列表
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer]
        optimizers = [opt for opt in optimizers if opt is not None]

        # 初始化损失和系数的列表
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        # 进行指定次数的梯度更新
        for _ in range(gradient_steps):
            # 从经验回放缓冲区中采样
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # 如果使用随机特征估计，则重置actor的噪声
            if self.use_sde:
                self.actor.reset_noise(self.batch_size)

            # 计算下一个状态的动作及其对数概率
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations, replay_data.task_ids)

                # 获取目标Critic网络对下一个状态动作对的Q值
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions, replay_data.task_ids), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # 计算目标Q值
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * (next_q_values - self.ent_coef_tensor * next_log_prob)

            # 获取当前Critic网络的Q值
            current_q_values = self.critic(replay_data.observations, replay_data.actions, replay_data.task_ids)
            # 计算Critic的损失
            critic_loss = sum(th.nn.functional.mse_loss(current_q, target_q_values) for current_q in current_q_values)

            # 计算Actor网络的动作和对数概率
            actor_actions, log_prob = self.actor.action_log_prob(replay_data.observations, replay_data.task_ids)
            q_values_pi = th.cat(self.critic(replay_data.observations, actor_actions, replay_data.task_ids), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            # 计算Actor的损失
            actor_loss = (self.ent_coef_tensor * log_prob - min_qf_pi).mean()

            # 如果存在熵系数优化器，则优化熵系数
            if self.ent_coef_optimizer is not None:
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                self.ent_coef_tensor = self.log_ent_coef.exp().detach()
                ent_coef_losses.append(ent_coef_loss.item())
                ent_coefs.append(self.ent_coef_tensor.item())

            # 记录损失
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

            # 更新Actor网络
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # 更新Critic网络
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

        # 记录训练的更新次数和损失
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/entropy_loss", np.mean(ent_coef_losses))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    # 返回在保存模型时不包含的参数列表
    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]

    # 返回需要保存的模型参数
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer", "critic_target"]
        return state_dicts, []
