# 截止 26.1.27 最新一版，比较五个模型存活时间



import os
import glob
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from common_utils import create_env, create_gym_env
from config import ENV_NAME, SAVE_PATH, OBS_ATTR_TO_KEEP, ACT_ATTR_TO_KEEP

# ==========================================
# 0. 用户控制面板
# ==========================================
TEST_SEED = 2024

# 【N-3 攻击目标】
TARGET_LINES = [2,10,19]
FAULT_START_STEP = 100

# 【自由开关】
MODELS_TO_TEST = {
    "Baseline": True,  # Do Nothing
    "Original PPO": True,  # 原始 PPO
    "Improved PPO": True,  # SOTA (带 PageRank)
    "A2C": True,  # A2C
    "DQN": True  # DQN
}

# --- 模型文件路径配置 ---
PATHS = {
    "Original PPO": {
        "model": os.path.join(SAVE_PATH, "ppo_pure_baseline_final.zip"),
        "norm": os.path.join(SAVE_PATH, "ppo_pure_baseline_vecnormalize.pkl")
    },
    "Improved PPO": {
        "model": r"C:\Users\lw\Desktop\0424-Dianli\0424-Dianli\code\saved_model\ppo_final_sota_final.zip",
        "norm": r"C:\Users\lw\Desktop\0424-Dianli\0424-Dianli\code\saved_model\ppo_final_sota_vecnormalize.pkl"
    },
    "A2C": {
        "model": os.path.join(SAVE_PATH, "a2c_grid2op_final.zip"),
        "norm": None
    },
    "DQN": {
        "model": os.path.join(SAVE_PATH, "dqn_grid2op_final.zip"),
        "norm": None
    }
}

B_MATRIX_PATH = r'D:\xuexi\dianwangjilian\matlabmodels\data\dc\IEEE14\causal_results_csv'


# ==========================================
# 1. 必要的 Wrappers (PageRank & Discrete)
# ==========================================
class PageRankRiskWrapper(gym.Wrapper):
    """用于 Improved PPO 的风险感知 Wrapper"""

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
            if files: return pd.read_csv(files[0]).shape[0]
        except:
            pass
        return 20

    def _load_b_matrix_cached(self, line_index):
        if line_index in self.b_matrix_cache: return self.b_matrix_cache[line_index]
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
        if not self.fault_chain: return np.zeros(self.num_lines, dtype=np.float32)
        try:
            most_recent = self.fault_chain[-1]
            B_base = self._load_b_matrix_cached(most_recent)
            B_mod = B_base.copy()
            for i in set(self.fault_chain[:-1]):
                if i < self.num_lines: B_mod[i, :] = 0
            B_mod[np.abs(B_mod) < self.threshold] = 0.0
            G = nx.from_numpy_array(np.abs(B_mod).T, create_using=nx.DiGraph)
            pers = dict.fromkeys(G.nodes, 0.0)
            if most_recent in G:
                pers[most_recent] = 1.0
            else:
                return np.zeros(self.num_lines)
            scores = nx.pagerank(G, alpha=self.damping, personalization=pers, weight='weight')
            vec = np.zeros(self.num_lines, dtype=np.float32)
            cands = set(range(self.num_lines)) - set(self.fault_chain)
            if cands:
                final = {n: s for n, s in scores.items() if n in cands}
                if sum(final.values()) > 0:
                    k = int(math.ceil(self.k_percent * self.num_lines))
                    for n, _ in sorted(final.items(), key=lambda x: x[1], reverse=True)[:k]: vec[n] = 1.0
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
        res = self.env.step(action)
        if len(res) == 5:
            obs, r, term, trunc, i = res;
            done = term or trunc
        else:
            obs, r, done, i = res;
            term, trunc = done, False
        try:
            curr = i.get('line_status', self.env.unwrapped.backend.get_line_status() if hasattr(self.env.unwrapped,
                                                                                                'backend') else self.last_line_status)
            if curr is not None:
                for l in np.where(self.last_line_status & ~curr)[0]:
                    if l not in self.fault_chain: self.fault_chain.append(l)
                self.last_line_status = curr.copy()
        except:
            pass
        return np.concatenate([obs, self._run_pagerank_logic()]), r, term, trunc, i


class DiscretizeActionWrapper(gym.Wrapper):
    """用于 DQN 的离散动作 Wrapper"""

    def __init__(self, env, n_actions_per_dim=3, max_actions=729):
        super().__init__(env)
        self.original_action_space = env.action_space
        self.low = self.original_action_space.low
        self.high = self.original_action_space.high
        self.action_dims = self.original_action_space.shape[0]
        self.n_actions_per_dim = n_actions_per_dim
        theoretical_actions = n_actions_per_dim ** self.action_dims
        self.n_discrete_actions = min(theoretical_actions, max_actions)
        self.action_space = spaces.Discrete(self.n_discrete_actions)
        self._create_action_map()

    def _create_action_map(self):
        self.action_map = {}
        action_idx = 0
        mid_action = np.mean([self.low, self.high], axis=0)
        self.action_map[action_idx] = mid_action
        action_idx += 1
        for dim in range(self.action_dims):
            for level in np.linspace(self.low[dim], self.high[dim], self.n_actions_per_dim):
                if action_idx >= self.n_discrete_actions: break
                action = mid_action.copy()
                action[dim] = level
                self.action_map[action_idx] = action
                action_idx += 1
        while action_idx < self.n_discrete_actions:
            self.action_map[action_idx] = np.random.uniform(self.low, self.high)
            action_idx += 1

    def step(self, action):
        continuous_action = self.action_map[action]
        return self.env.step(continuous_action)


# ==========================================
# 2. 环境工厂
# ==========================================
def make_env_for_model(algo_name):
    # 1. 基础环境
    env_native = create_env(ENV_NAME)
    env = create_gym_env(env_native, OBS_ATTR_TO_KEEP, ACT_ATTR_TO_KEEP)

    # 2. 根据算法挂载 Wrapper
    if algo_name == "Improved PPO":
        print(f"  -> Adding PageRankRiskWrapper for {algo_name}")
        env = PageRankRiskWrapper(env, b_matrix_path=B_MATRIX_PATH)
    elif algo_name == "DQN":
        print(f"  -> Adding Flatten & Discretize Wrapper for {algo_name}")
        env = FlattenObservation(env)
        env = DiscretizeActionWrapper(env, n_actions_per_dim=3)

    # 3. 包装成 VecEnv
    env = DummyVecEnv([lambda: env])
    return env


# ==========================================
# 3. 核心评估循环 (修复版)
# ==========================================
def get_grid2op_env(env):
    """提取底层 Grid2Op 环境"""
    current = env
    # 尝试剥离 gym wrappers (VecEnv -> Monitor -> etc -> Grid2OpEnv)
    if hasattr(current, 'envs'):
        current = current.envs[0]
    while hasattr(current, 'unwrapped') and current != current.unwrapped:
        current = current.unwrapped
    # 尝试获取 init_env (Grid2Op 原始环境通常藏在这里)
    if hasattr(current, 'init_env'):
        current = current.init_env
    return current


def run_episode(env, model=None, name="Agent", seed=42, max_steps=2000):
    print(f"Running simulation for: {name}...")
    real_env = get_grid2op_env(env)

    try:
        env.seed(seed);
        real_env.seed(seed)
    except:
        pass

    # 适配 gym 版本差异
    try:
        reset_res = env.reset(seed=seed)
    except:
        reset_res = env.reset()

    if isinstance(reset_res, tuple):
        obs = reset_res[0]
    else:
        obs = reset_res

    pending_outages = list(TARGET_LINES)
    done = False
    rho_history = []
    status_history = []
    step_count = 0

    while not done and step_count < max_steps:
        step_count += 1

        # --- 1. 记录数据 ---
        try:
            curr_rho = real_env.backend.get_relative_flow()
            curr_status = real_env.backend.get_line_status()
        except:
            curr_rho = np.zeros(20)
            curr_status = np.ones(20, dtype=bool)
        rho_history.append(curr_rho)
        status_history.append(curr_status)

        # --- 2. 故障注入 (Sabotage) ---
        if step_count >= FAULT_START_STEP and len(pending_outages) > 0:
            target_line = pending_outages.pop(0)
            try:
                sabotage_action = real_env.action_space({"set_line_status": [(target_line, -1)]})
                # 注意：这里我们只关心 sabotage 是否直接搞挂了环境
                _, _, d, info_sabotage = real_env.step(sabotage_action)

                if d:
                    done = True
                    print(f"  [Step {step_count}] SYSTEM COLLAPSED during sabotage!")
                    if 'exception' in info_sabotage:
                        print(f"  -> 原因: {info_sabotage['exception']}")
                    continue

                if not real_env.backend.get_line_status()[target_line]:
                    print(f"  [Step {step_count}] Sabotage SUCCESS: Line {target_line} DOWN.")
                else:
                    print(f"  [Step {step_count}] Sabotage FAILED: Line {target_line} UP.")

            except Exception as e:
                print(f"    -> ERROR in Sabotage: {e}")

        # --- 3. Agent Action ---
        info = {}
        if model is None:  # Baseline
            try:
                action = real_env.action_space({})
                obs, reward, done, info = real_env.step(action)
            except Exception as e:
                # Baseline 经常因为无作为导致发散，这里捕获一下
                print(f"  Baseline Terminated: {e}")
                done = True
        else:  # AI Models
            action, _ = model.predict(obs, deterministic=True)
            res = env.step(action)

            if len(res) == 5:
                obs, reward, term, trunc, info = res
                done = term or trunc
            else:
                obs, reward, done, info = res

        # --- 4. 打印死亡原因分析 (DQN 诊断核心) ---
        if done:
            print(f"  [STOP] Episode terminated at step {step_count}")

            # 兼容 VecEnv 的 info 是个列表的情况
            if isinstance(info, list):
                if len(info) > 0:
                    info = info[0]
                else:
                    info = {}

            # 情况 A: 物理求解发散
            if 'exception' in info and info['exception']:
                print(f"  -> 🔴 物理发散 (Divergence/Solver Error):")
                print(f"     {info['exception']}")

            # 情况 B: 非法动作
            elif 'is_illegal' in info and info['is_illegal']:
                print(f"  -> 🚫 非法动作 (Illegal Action): Agent 尝试了无效操作")

            # 情况 C: 游戏结束代码
            elif 'game_over' in info:
                print(f"  -> ⚡ 游戏结束 (Game Over Code): {info.get('game_over')}")

        if step_count % 500 == 0: print(f"  ... Step {step_count}")

    print(f"  -> Finished. Steps: {step_count}")
    return np.array(rho_history), np.array(status_history), step_count


# ==========================================
# 4. 可视化函数
# ==========================================
def plot_results(results, save_path):
    n_plots = len(results)
    if n_plots == 0: return

    plt.figure(figsize=(24, 6 * n_plots))
    sns.set_style("whitegrid")
    cmap = plt.get_cmap('tab20')

    first_res = list(results.values())[0]
    num_lines = first_res[0].shape[1] if len(first_res[0]) > 0 else 20
    all_lines = range(num_lines)
    line_colors = {i: cmap(i % 20) for i in all_lines}

    plot_idx = 1
    sort_order = ["Baseline", "Original PPO", "Improved PPO", "A2C", "DQN"]
    sorted_names = [n for n in sort_order if n in results]

    for name in sorted_names:
        rho_data, status_data, steps = results[name]
        ax = plt.subplot(n_plots, 1, plot_idx)
        plot_idx += 1

        if len(rho_data) == 0:
            ax.text(0.5, 0.5, "No Data", ha='center')
            continue

        # 画线
        for line in all_lines:
            y_values = rho_data[:, line]
            color = line_colors[line]
            label = f"Line {line}"
            width = 1.5;
            linestyle = '-';
            alpha = 0.8
            if line in TARGET_LINES:
                width = 2.5;
                linestyle = '--';
                label = f"Line {line} (Target)";
                alpha = 1.0
            ax.plot(y_values, color=color, alpha=alpha, linewidth=width, linestyle=linestyle, label=label)

        # 画故障点
        limit = min(len(status_data), len(rho_data))
        marked_events = []
        for t in range(1, limit):
            just_tripped = np.where(status_data[t - 1] & ~status_data[t])[0]
            for line_idx in just_tripped:
                is_initial = (line_idx in TARGET_LINES) and (t < FAULT_START_STEP + 5)
                y_pos = min(rho_data[t - 1, line_idx], 2.0)
                marker = 'v' if is_initial else 'X'
                mcolor = 'black' if is_initial else 'red'
                msize = 80 if is_initial else 150
                txt = "Attack" if is_initial else f"Trip L{line_idx}"
                ax.scatter(t, y_pos, marker=marker, color=mcolor, s=msize, zorder=10, edgecolor='white')

                if steps < 200 or is_initial:
                    y_offset = 15
                    for (mt, my) in marked_events:
                        if abs(mt - t) < 5 and abs(my - y_pos) < 0.2: y_offset += 20
                    marked_events.append((t, y_pos))
                    ax.annotate(txt, (t, y_pos), xytext=(5, y_offset), textcoords='offset points',
                                fontsize=9, color=mcolor, fontweight='bold')

        # 装饰
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Limit')
        ax.axvline(FAULT_START_STEP, color='black', linestyle=':', label='Sabotage Start')
        title_color = '#2CA02C' if steps > 500 else '#D62728'
        status_txt = "Stabilized" if steps > 500 else "Collapse"
        ax.set_title(f"{name} | {status_txt} (Steps: {steps})", fontsize=16, fontweight='bold', color=title_color)
        ax.set_ylabel("Load Ratio")
        if plot_idx == n_plots + 1: ax.set_xlabel("Time Steps")

        handles, labels = ax.get_legend_handles_labels()

        def sort_key(item):
            if "Limit" in item[1]: return 0
            if "Sabotage" in item[1]: return 1
            if "Target" in item[1]: return 2
            return 3

        sorted_legs = sorted(zip(handles, labels), key=sort_key)
        sh, sl = zip(*sorted_legs)
        ax.legend(sh, sl, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='small', ncol=1)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot saved to: {save_path}")


# ==========================================
# 5. 主流程
# ==========================================
def main():
    print(f"Starting Multi-Model Evaluation (N-3 Attack: {TARGET_LINES})")
    results = {}

    for algo_name, is_enabled in MODELS_TO_TEST.items():
        if not is_enabled: continue

        print(f"\n" + "=" * 40)
        print(f"Testing: {algo_name}")
        print("=" * 40)

        # 1. 准备环境
        env = make_env_for_model(algo_name)
        model = None

        # 2. 加载模型
        if algo_name != "Baseline":
            cfg = PATHS.get(algo_name, {})
            model_path = cfg.get("model")
            norm_path = cfg.get("norm")

            if model_path and os.path.exists(model_path):
                if norm_path and os.path.exists(norm_path):
                    print(f"  Loading stats from {norm_path}")
                    env = VecNormalize.load(norm_path, env)
                    env.training = False;
                    env.norm_reward = False
                elif algo_name in ["Original PPO", "Improved PPO"]:
                    print("  [Warning] PPO stats not found! Using auto-adapt.")
                    env = VecNormalize(env, norm_obs=True, norm_reward=False, training=True, clip_obs=10.)

                try:
                    if "PPO" in algo_name:
                        model = PPO.load(model_path)
                    elif algo_name == "A2C":
                        model = A2C.load(model_path)
                    elif algo_name == "DQN":
                        model = DQN.load(model_path)
                    print(f"  Model loaded successfully.")
                except Exception as e:
                    print(f"  [Error] Failed to load model: {e}")
                    env.close();
                    continue
            else:
                print(f"  [Error] Model file not found: {model_path}")
                env.close();
                continue

        # 3. 运行仿真
        rho, status, steps = run_episode(env, model, name=algo_name, seed=TEST_SEED)
        results[algo_name] = (rho, status, steps)
        env.close()

    # 4. 汇总
    print("\n" + "=" * 40)
    print("FINAL SUMMARY")
    print("=" * 40)
    for name, (_, _, steps) in results.items():
        print(f"{name:<15}: {steps} steps")

    plot_results(results, os.path.join(SAVE_PATH, "evaluation_results", "multi_model_comparison.png"))


if __name__ == "__main__":
    main()