import wandb
import os
import glob
import math
import numpy as np
import pandas as pd
import networkx as nx
import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# 导入项目工具
from common_utils import create_env, create_gym_env, get_common_training_params
from config import *

# ==========================================
# 0. 环境设置
# ==========================================
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["WANDB_INSECURE_DISABLE_SSL"] = "true"


# ==========================================
# 1. PageRank Wrapper (因果推断核心)
# ==========================================
class PageRankRiskWrapper(gym.Wrapper):
    """
    将基于因果图 (B-Matrix) 的 PageRank 风险评分拼接到观测空间中。
    """

    def __init__(self, env, b_matrix_path, k_percent=0.80, damping=0.85, threshold=0.05):
        super().__init__(env)
        self.b_matrix_path = b_matrix_path
        self.k_percent = k_percent
        self.damping = damping
        self.threshold = threshold
        self.num_lines = self._detect_num_lines()
        self.b_matrix_cache = {}
        self.last_line_status = None
        self.fault_chain = []

        orig_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(orig_dim + self.num_lines,), dtype=np.float32
        )
        self.action_space = env.action_space

    def _detect_num_lines(self):
        try:
            search_pattern = os.path.join(self.b_matrix_path, "causal_B_anomaly_F*.csv")
            files = glob.glob(search_pattern)
            if files:
                return pd.read_csv(files[0]).shape[0]
        except:
            pass
        return 20

    def _load_b_matrix_cached(self, line_index):
        if line_index in self.b_matrix_cache:
            return self.b_matrix_cache[line_index]
        try:
            pattern = os.path.join(self.b_matrix_path, f"causal_B_anomaly_F{line_index:04d}*.csv")
            files = glob.glob(pattern)
            if files:
                mat = pd.read_csv(files[0], header=None).to_numpy()
                if mat.shape == (self.num_lines, self.num_lines):
                    self.b_matrix_cache[line_index] = mat
                    return mat
        except:
            pass
        return np.zeros((self.num_lines, self.num_lines))

    def _run_pagerank_logic(self):
        if not self.fault_chain:
            return np.zeros(self.num_lines, dtype=np.float32)
        try:
            most_recent_fault = self.fault_chain[-1]
            B_base = self._load_b_matrix_cached(most_recent_fault)
            B_mod = B_base.copy()

            # Intervention: 移除已知故障的影响
            prev_faults = set(self.fault_chain[:-1])
            for i in prev_faults:
                if i < self.num_lines: B_mod[i, :] = 0

            B_mod[np.abs(B_mod) < self.threshold] = 0.0

            G = nx.from_numpy_array(np.abs(B_mod).T, create_using=nx.DiGraph)
            pers = dict.fromkeys(G.nodes, 0.0)
            if most_recent_fault in G:
                pers[most_recent_fault] = 1.0
            else:
                return np.zeros(self.num_lines)

            scores = nx.pagerank(G, alpha=self.damping, personalization=pers, weight='weight')
            vec = np.zeros(self.num_lines, dtype=np.float32)
            candidates = set(range(self.num_lines)) - set(self.fault_chain)

            if candidates:
                final_scores = {n: s for n, s in scores.items() if n in candidates}
                if sum(final_scores.values()) > 0:
                    k = int(math.ceil(self.k_percent * self.num_lines))
                    top_k = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:k]
                    for node_idx, _ in top_k:
                        vec[node_idx] = 1.0
            return vec
        except:
            return np.zeros(self.num_lines)

    def reset(self, seed=None, options=None):
        try:
            obs, info = self.env.reset(seed=seed, options=options)
        except:
            obs = self.env.reset()
            info = {}
        self.last_line_status = np.ones(self.num_lines, dtype=bool)
        self.fault_chain = []
        return np.concatenate([obs, np.zeros(self.num_lines, dtype=np.float32)]), info

    def step(self, action):
        step_res = self.env.step(action)
        if len(step_res) == 5:
            obs, reward, terminated, truncated, info = step_res
            done = terminated or truncated
        else:
            obs, reward, done, info = step_res
            terminated, truncated = done, False

        try:
            if 'line_status' in info:
                curr_status = info['line_status']
            elif hasattr(self.env.unwrapped, 'backend'):
                curr_status = self.env.unwrapped.backend.get_line_status()
            else:
                curr_status = self.last_line_status

            if curr_status is not None:
                new_disconnected = np.where(self.last_line_status & ~curr_status)[0]
                for line_idx in new_disconnected:
                    if line_idx not in self.fault_chain:
                        self.fault_chain.append(line_idx)
                self.last_line_status = curr_status.copy()
        except:
            pass

        risk_pred = self._run_pagerank_logic()
        new_obs = np.concatenate([obs, risk_pred])

        if len(step_res) == 5:
            return new_obs, reward, terminated, truncated, info
        else:
            return new_obs, reward, done, info


# ==========================================
# 2. WandB 回调函数 (已恢复您的自定义参数！)
# ==========================================
class WandbMetricsCallback(BaseCallback):
    """
    完全恢复了您定义的 load_loss, action_efficiency 以及 custom/ 前缀日志。
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metrics = {}
        # 初始化最优值记录
        self.best_metrics = {
            'reward': float('-inf'),
            'recovery_steps': float('-inf'),
            'action_efficiency': float('-inf')
        }

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        # 1. 获取本轮数据
        if len(self.model.ep_info_buffer) > 0:
            rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            lengths = [ep_info["l"] for ep_info in self.model.ep_info_buffer]

            # 2. 计算基础指标
            self.metrics['mean_reward'] = np.mean(rewards)
            self.metrics['best_reward'] = max(self.best_metrics['reward'], np.max(rewards) if rewards else -999)
            self.metrics['mean_length'] = np.mean(lengths)
            self.metrics['recovery_steps'] = int(np.mean(lengths))

            # 3. 计算效率指标
            if self.metrics['recovery_steps'] > 0:
                self.metrics['action_efficiency'] = self.metrics['mean_reward'] / self.metrics['recovery_steps']
            else:
                self.metrics['action_efficiency'] = 0.0

            # 4. 计算 Load Loss (按您的逻辑取反)
            self.metrics['load_loss'] = -self.metrics['mean_reward']

            # 5. 获取 PPO 内部 Loss
            if hasattr(self.model, 'logger'):
                logs = self.model.logger.name_to_value
                self.metrics['policy_loss'] = logs.get("train/policy_loss", 0)
                self.metrics['value_loss'] = logs.get("train/value_loss", 0)
                self.metrics['entropy'] = logs.get("train/entropy", 0)

            # 6. 更新历史最优
            if self.metrics['mean_reward'] > self.best_metrics['reward']:
                self.best_metrics['reward'] = self.metrics['mean_reward']
            if self.metrics['recovery_steps'] > self.best_metrics['recovery_steps']:
                self.best_metrics['recovery_steps'] = self.metrics['recovery_steps']
            if self.metrics['action_efficiency'] > self.best_metrics['action_efficiency']:
                self.best_metrics['action_efficiency'] = self.metrics['action_efficiency']

            # 7. 【关键】上传完整日志到 WandB
            if wandb.run is not None:
                wandb.log({
                    # 您的自定义业务指标
                    'custom/mean_reward': float(self.metrics.get('mean_reward', 0)),
                    'custom/best_reward': float(self.metrics.get('best_reward', 0)),
                    'custom/mean_length': float(self.metrics.get('mean_length', 0)),
                    'custom/load_loss': float(self.metrics.get('load_loss', 0)),
                    'custom/recovery_steps': int(self.metrics.get('recovery_steps', 0)),
                    'custom/action_efficiency': float(self.metrics.get('action_efficiency', 0)),

                    # 训练 Loss
                    'train/policy_loss': float(self.metrics.get('policy_loss', 0)),
                    'train/value_loss': float(self.metrics.get('value_loss', 0)),
                    'train/entropy': float(self.metrics.get('entropy', 0)),

                    # 用于 Sweep 优化的核心指标 (无前缀，方便配置)
                    'recovery_steps': int(self.metrics.get('recovery_steps', 0)),
                    'mean_reward': float(self.metrics.get('mean_reward', 0)),

                    # 历史最优记录
                    'best/best_metric_reward': self.best_metrics['reward'],
                    'best/best_metric_recovery': self.best_metrics['recovery_steps'],

                    'global_step': self.num_timesteps
                })

            if self.verbose > 0:
                print(
                    f"Step {self.num_timesteps}: Reward={self.metrics['mean_reward']:.2f}, Survival={self.metrics['recovery_steps']}")


# ==========================================
# 3. Sweep 训练主函数
# ==========================================
def train_sweep_func(config=None):
    with wandb.init(config=config, sync_tensorboard=True) as run:
        config = wandb.config

        # 请确保路径正确
        B_MATRIX_PATH = r'D:\xuexi\dianwangjilian\matlabmodels\data\dc\IEEE14\causal_results_csv'
        if not os.path.exists(B_MATRIX_PATH):
            print(f"Warning: B_Matrix path {B_MATRIX_PATH} not found.")

        print(
            f"--> Sweep Run {run.id}: PR_K={config.pr_k_percent:.2f}, LR={config.learning_rate:.6f}, N_Steps={config.n_steps}")

        env = create_env(ENV_NAME)

        try:
            env_gym = create_gym_env(env, OBS_ATTR_TO_KEEP, ACT_ATTR_TO_KEEP)

            # 业务 Wrapper
            env_gym = PageRankRiskWrapper(
                env_gym,
                b_matrix_path=B_MATRIX_PATH,
                k_percent=config.pr_k_percent,
                damping=config.pr_damping,
                threshold=0.05
            )

            # 向量化 & 归一化 (注意顺序: Monitor -> Normalize)
            env_vec = DummyVecEnv([lambda: env_gym])
            env_vec = VecMonitor(env_vec)
            env_vec = VecNormalize(env_vec, norm_obs=True, norm_reward=True, clip_obs=10.)

            # 参数配置
            common_params = get_common_training_params()
            current_params = common_params.copy()
            current_params.update({
                'learning_rate': config.learning_rate,
                'n_steps': config.n_steps,
                'batch_size': config.batch_size,
                'ent_coef': config.ent_coef,
            })

            # 模型初始化
            model = PPO(
                policy="MlpPolicy",
                env=env_vec,
                device="cuda",
                tensorboard_log=f"runs/{run.id}",
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=nn.Tanh),
                **current_params
            )

            # 训练 (Sweep 阶段跑 30万步)
            SWEEP_TIMESTEPS = 300_000

            # 使用增强版 Callback
            model.learn(
                total_timesteps=SWEEP_TIMESTEPS,
                callback=WandbMetricsCallback(verbose=1),
                progress_bar=False
            )

            # 保存
            # os.makedirs(SAVE_PATH, exist_ok=True)
            # model.save(os.path.join(SAVE_PATH, f"model_sweep_{run.id}.zip"))

        except Exception as e:
            print(f"Run {run.id} Failed with error: {e}")
        finally:
            env.close()


# ==========================================
# 4. Sweep Configuration
# ==========================================
sweep_configuration = {
    'method': 'bayes',
    'name': 'ppo-causal-grid2op',
    'metric': {
        'goal': 'maximize',
        'name': 'recovery_steps'  # 这里对应 WandBMetricsCallback log 里的 key
    },
    'parameters': {
        'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 3e-4},
        'n_steps': {'values': [512, 1024, 2048]},
        'batch_size': {'values': [64, 128]},
        'ent_coef': {'distribution': 'uniform', 'min': 0.0, 'max': 0.01},
        'pr_k_percent': {'distribution': 'uniform', 'min': 0.6, 'max': 0.95},
        'pr_damping': {'values': [0.5, 0.85]}
    }
}

if __name__ == "__main__":
    os.makedirs("runs", exist_ok=True)
    sweep_id = wandb.sweep(sweep_configuration, project='ppo-grid2op-causal')
    print(f"Sweep ID: {sweep_id}")
    wandb.agent(sweep_id, function=train_sweep_func, count=20)
    print("Sweep Finished.")