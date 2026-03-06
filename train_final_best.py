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
from stable_baselines3.common.callbacks import CheckpointCallback

# 导入工具
from common_utils import create_env, create_gym_env
from config import ENV_NAME, SAVE_PATH, OBS_ATTR_TO_KEEP, ACT_ATTR_TO_KEEP

# ==========================================
# 1. 最佳参数配置 (来自 WandB)
# ==========================================
BEST_PARAMS = {
    # PPO 参数
    "learning_rate": 0.00006020043631749061,
    "n_steps": 2048,
    "batch_size": 128,
    "ent_coef": 0.007910756762737736,

    # 论文核心参数 (PageRank)
    "pr_k_percent": 0.9110186501048008,
    "pr_damping": 0.85,
}

# 训练配置
TOTAL_TIMESTEPS = 2_000_000  # 200万步，确保充分收敛
EXP_NAME = "ppo_final_sota"  # 实验名称
B_MATRIX_PATH = r'D:\xuexi\dianwangjilian\matlabmodels\data\dc\IEEE14\causal_results_csv'


# ==========================================
# 2. PageRank Wrapper (完整逻辑版)
# ==========================================
class PageRankRiskWrapper(gym.Wrapper):
    """
    将基于因果图 (B-Matrix) 的 PageRank 风险评分拼接到观测空间中。
    包含完整的读取、缓存和计算逻辑。
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

        # 扩展观测空间
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

            # 干预: 移除已知故障的影响
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
            obs = self.env.reset();
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
# 3. 训练主程序
# ==========================================
def main():
    print(f"🚀 启动最终 SOTA 训练")
    print(f"目标步数: {TOTAL_TIMESTEPS}")
    print(f"参数配置: {BEST_PARAMS}")

    os.makedirs(SAVE_PATH, exist_ok=True)

    # 1. 环境
    env = create_env(ENV_NAME)
    env = create_gym_env(env, OBS_ATTR_TO_KEEP, ACT_ATTR_TO_KEEP)

    # 2. 【关键】加上 PageRank Wrapper
    print("正在加载因果推断模块 (PageRank)...")
    env = PageRankRiskWrapper(
        env,
        b_matrix_path=B_MATRIX_PATH,
        k_percent=BEST_PARAMS['pr_k_percent'],
        damping=BEST_PARAMS['pr_damping']
    )

    # 3. 标准化 (重要：训练完会自动保存统计数据)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 4. 模型
    model = PPO(
        "MlpPolicy",
        env,
        device="cuda",
        verbose=1,
        tensorboard_log=f"./runs/{EXP_NAME}",
        learning_rate=BEST_PARAMS['learning_rate'],
        n_steps=BEST_PARAMS['n_steps'],
        batch_size=BEST_PARAMS['batch_size'],
        ent_coef=BEST_PARAMS['ent_coef'],
        policy_kwargs=dict(net_arch=[256, 256], activation_fn=nn.Tanh)
    )

    # 5. 自动保存 Checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=os.path.join(SAVE_PATH, "checkpoints"),
        name_prefix=EXP_NAME
    )

    # 6. 开始训练
    print("开始训练... (这可能需要几小时)")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, progress_bar=True)

    # 7. 保存最终结果
    final_model_path = os.path.join(SAVE_PATH, f"{EXP_NAME}_final.zip")
    final_vec_path = os.path.join(SAVE_PATH, f"{EXP_NAME}_vecnormalize.pkl")

    model.save(final_model_path)
    env.save(final_vec_path)  # 保存 VecNormalize 统计量，评估时必须用！

    print("\n✅ 训练结束！")
    print(f"模型已保存: {final_model_path}")
    print(f"归一化参数已保存: {final_vec_path}")


if __name__ == "__main__":
    main()