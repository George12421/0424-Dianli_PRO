import grid2op
from grid2op.Reward import LinesCapacityReward
from grid2op.Chronics import MultifolderWithCache
from lightsim2grid import LightSimBackend
from grid2op.gym_compat import BoxGymObsSpace, BoxGymActSpace, GymEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import re
import os
import json
import time
from config import SAVE_PATH, PPO_MODEL_NAME
import wandb


class MetricsCallback(BaseCallback):
    def __init__(self, model_name=PPO_MODEL_NAME, verbose=0):
        super().__init__(verbose)
        self.metrics = {}
        self.model_name = model_name
        self.best_metrics = {
            'reward': float('-inf'),
            'recovery_steps': float('-inf'),  # 修正：存活步数越大越好，初始化为负无穷
            'action_efficiency': float('-inf')
        }

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if len(self.model.ep_info_buffer) > 0:
            rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            lengths = [ep_info["l"] for ep_info in self.model.ep_info_buffer]

            # --- 1. 计算指标 ---
            self.metrics['mean_reward'] = np.mean(rewards)
            self.metrics['best_reward'] = max(self.best_metrics['reward'], self.metrics['mean_reward'])
            self.metrics['mean_length'] = np.mean(lengths)
            self.metrics['recovery_steps'] = int(np.mean(lengths))

            if self.metrics['recovery_steps'] > 0:
                self.metrics['action_efficiency'] = self.metrics['mean_reward'] / self.metrics['recovery_steps']
            else:
                self.metrics['action_efficiency'] = 0.0

            self.metrics['load_loss'] = -self.metrics['mean_reward']

            # 获取网络损失
            if hasattr(self.model, 'logger'):
                logs = self.model.logger.name_to_value
                self.metrics['policy_loss'] = logs.get("train/policy_loss", 0)
                self.metrics['value_loss'] = logs.get("train/value_loss", 0)
                self.metrics['entropy'] = logs.get("train/entropy", 0)

            # --- 2. 上传到 WandB ---
            if wandb.run is not None:
                wandb.log({
                    'custom/mean_reward': float(self.metrics.get('mean_reward', 0)),
                    'custom/recovery_steps': int(self.metrics.get('recovery_steps', 0)),
                    'custom/action_efficiency': float(self.metrics.get('action_efficiency', 0)),
                    'train/policy_loss': float(self.metrics.get('policy_loss', 0)),
                    'train/value_loss': float(self.metrics.get('value_loss', 0)),
                    'train/entropy': float(self.metrics.get('entropy', 0)),
                    'global_step': self.num_timesteps
                })

            # --- 3. 检查并保存模型 ---
            self._check_and_save_model()

    def _check_and_save_model(self):
        """检查各个指标是否达到最优，如果是则保存模型"""
        # 1. 最佳奖励
        if self.metrics['mean_reward'] > self.best_metrics['reward']:
            self.best_metrics['reward'] = self.metrics['mean_reward']
            self._save_model('best_reward')

        # 2. 最佳存活 (越长越好)
        if self.metrics.get('recovery_steps', 0) > self.best_metrics['recovery_steps']:
            self.best_metrics['recovery_steps'] = self.metrics['recovery_steps']
            self._save_model('best_recovery_steps')

        # 3. 最佳效率
        if self.metrics.get('action_efficiency', float('-inf')) > self.best_metrics['action_efficiency']:
            self.best_metrics['action_efficiency'] = self.metrics['action_efficiency']
            self._save_model('best_action_efficiency')

    def _save_model(self, metric_name):
        save_path = os.path.join(SAVE_PATH, f"{self.model_name}_{metric_name}")
        self.model.save(save_path)

        # 同时保存一份 JSON
        metrics_path = os.path.join(SAVE_PATH, f"{self.model_name}_{metric_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, default=str, indent=4)

        print(
            f"已保存 {metric_name} 模型: Reward={self.metrics['mean_reward']:.2f}, Steps={self.metrics['recovery_steps']}")


def get_common_training_params():
    """
    【重要修正】
    这里改为实战参数，防止单独调用时 AI 变成弱智。
    Sweep 脚本会用自己的参数覆盖这里，所以不用担心冲突。
    """
    return {
        "n_steps": 2048,  # 给 AI 足够长的视野
        "batch_size": 64,  # 稳定的梯度更新
        "n_epochs": 10,  # 充分利用数据
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "target_kl": None,
    }


def create_env(env_name="l2rpn_case14_sandbox",
               reward_class=LinesCapacityReward,
               backend=None,
               chronics_class=MultifolderWithCache):
    if backend is None:
        backend = LightSimBackend()

    env = grid2op.make(env_name,
                       reward_class=reward_class,
                       backend=backend,
                       chronics_class=chronics_class)

    # 过滤数据，只用以 '00' 结尾的场景
    env.chronics_handler.real_data.set_filter(lambda x: re.match(".*00$", x) is not None)
    env.chronics_handler.real_data.reset()

    return env

def get_a2c_training_params():
    """Get training parameters specific to A2C algorithm."""
    return {
        "n_steps": 32,  # Drastically reduced for speed
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
    }

def create_gym_env(env, obs_attr_to_keep=None, act_attr_to_keep=None):
    if obs_attr_to_keep is None:
        # 这里保留了 rho，这对于你的论文实现至关重要
        obs_attr_to_keep = ["day_of_week", "hour_of_day", "minute_of_hour",
                            "prod_p", "prod_v", "load_p", "load_q",
                            "actual_dispatch", "target_dispatch", "topo_vect",
                            "time_before_cooldown_line", "time_before_cooldown_sub",
                            "rho", "timestep_overflow", "line_status",
                            "storage_power", "storage_charge"]

    if act_attr_to_keep is None:
        act_attr_to_keep = ["redispatch", "curtail", "set_storage"]

    env_gym = GymEnv(env)
    env_gym.observation_space.close()
    env_gym.observation_space = BoxGymObsSpace(env.observation_space,
                                               attr_to_keep=obs_attr_to_keep)
    env_gym.action_space.close()
    env_gym.action_space = BoxGymActSpace(env.action_space,
                                          attr_to_keep=act_attr_to_keep)

    return env_gym