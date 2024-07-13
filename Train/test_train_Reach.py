from stable_baselines3 import SAC
from Algorithm.pacosac import CustomSAC
from stable_baselines3.common.callbacks import BaseCallback
from ParameterTask.ReachTask import ReachHandlingEnv
from robopal.commons.gym_wrapper import GymWrapper

TRAIN = 1

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, log_dir, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir

    def _on_step(self) -> bool:
        if self.n_calls % 51200 == 0:
            self.model.save(self.log_dir + f"/model_saved/SAC/policy_{self.n_calls}")
        return True


log_dir = "../log/ReachTask"

if TRAIN:
    env = ReachHandlingEnv(render_mode='human')
    # env = ConveyorHandlingEnv(render_mode=None)
else:
    env = ReachHandlingEnv(render_mode='human')
env = GymWrapper(env)

# Initialize the model
# model = SAC(
#     'MlpPolicy',
#     env,
#     verbose=1,
#     tensorboard_log=log_dir,
# )

model = CustomSAC(
    'MlpPolicy',
    env,
    verbose=1,
    tensorboard_log=log_dir,
)

if TRAIN:
    # Train the model
    model.learn(int(5e6), callback=TensorboardCallback(log_dir=log_dir))
    model.save(log_dir + f"/Final")

else:
# Test the model
    model = SAC.load(log_dir + f"/model_saved/SAC/policy_102400")
    obs, info = env.reset()
    for i in range(int(1e6)):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
        print(f"Step: {i}, Reward: {reward}, Info: {info}")
    env.close()
