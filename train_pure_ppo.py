import os
import numpy as np
import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# 导入工具
from common_utils import create_env, create_gym_env
from config import ENV_NAME, SAVE_PATH, OBS_ATTR_TO_KEEP, ACT_ATTR_TO_KEEP

# ==========================================
# 1. 参数配置 (原始 PPO)
# ==========================================
BEST_PARAMS = {
    # PPO 核心参数
    "learning_rate": 0.00006020043631749061,
    "n_steps": 2048,
    "batch_size": 128,
    "ent_coef": 0.007910756762737736,
}

# 训练配置
TOTAL_TIMESTEPS = 2_000_000  # 200万步
EXP_NAME = "ppo_pure_baseline"  # 修改实验名称为纯净版 Baseline

# ==========================================
# 2. 训练主程序
# ==========================================
def main():
    print(f"🚀 启动原始 PPO (Baseline) 训练")
    print(f"目标步数: {TOTAL_TIMESTEPS}")
    print(f"参数配置: {BEST_PARAMS}")

    os.makedirs(SAVE_PATH, exist_ok=True)

    # 1. 创建环境
    env = create_env(ENV_NAME)
    env = create_gym_env(env, OBS_ATTR_TO_KEEP, ACT_ATTR_TO_KEEP)

    # 2. 标准化处理 (Standard PPO Pipeline)
    # 使用 DummyVecEnv 包装环境，并应用 VecNormalize 进行观测值和奖励的归一化
    # 这是 PPO 算法收敛的关键步骤
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. 初始化模型
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

    # 4. 自动保存 Checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=os.path.join(SAVE_PATH, "checkpoints"),
        name_prefix=EXP_NAME
    )

    # 5. 开始训练
    print("开始训练... (这可能需要几小时)")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, progress_bar=True)

    # 6. 保存最终结果
    final_model_path = os.path.join(SAVE_PATH, f"{EXP_NAME}_final.zip")
    final_vec_path = os.path.join(SAVE_PATH, f"{EXP_NAME}_vecnormalize.pkl")

    model.save(final_model_path)
    env.save(final_vec_path)  # 保存 VecNormalize 统计量，评估时必须用！

    print("\n✅ 训练结束！")
    print(f"模型已保存: {final_model_path}")
    print(f"归一化参数已保存: {final_vec_path}")


if __name__ == "__main__":
    main()