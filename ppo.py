import os
import json
from tqdm import tqdm
from stable_baselines3 import PPO
from common_utils import create_env, create_gym_env, MetricsCallback, get_common_training_params
from config import *
import torch.nn as nn
import time
import torch

# print(torch.__version__)
# print(f"CUDA Available: {torch.cuda.is_available()}")
# print(f"Device Count: {torch.cuda.device_count()}")
# if torch.cuda.is_available():
#     print(f"Current Device: {torch.cuda.get_device_name(0)}")


def train_ppo(use_tensorboard=False):
    """Train a PPO agent on the grid2op environment."""
    # Create environment
    env = create_env(ENV_NAME)
    
    try:
        # Create gym environment
        env_gym = create_gym_env(env, OBS_ATTR_TO_KEEP, ACT_ATTR_TO_KEEP)
        
        # Get common training parameters
        common_params = get_common_training_params()
        
        # Create PPO model with highly optimized parameters for speed
        model = PPO(
            policy="MlpPolicy",
            env=env_gym,
            device="cuda",  # Force CPU usage for better performance
            tensorboard_log=LOGS_DIR if use_tensorboard else None,
            policy_kwargs=dict(
                net_arch=NET_ARCH,
                activation_fn=nn.ReLU
            ),
            **common_params
        )
        
        # Create callback for metrics
        metrics_callback = MetricsCallback(model_name=PPO_MODEL_NAME)
        
        # Create progress bar
        pbar = tqdm(total=ITERATIONS, desc="Training PPO", position=0, leave=True)
        
        # Training loop with minimal steps
        for i in range(ITERATIONS):
            model.learn(total_timesteps=32, reset_num_timesteps=False, callback=metrics_callback)
            
            # Update progress bar with metrics
            if metrics_callback.metrics:
                pbar.set_postfix({
                    'mean_reward': f'{metrics_callback.metrics.get("mean_reward", 0):.2f}',
                    'best_reward': f'{metrics_callback.metrics.get("best_reward", 0):.2f}',
                    'mean_length': f'{metrics_callback.metrics.get("mean_length", 0):.2f}',
                    'load_loss': f'{metrics_callback.metrics.get("load_loss", 0):.2f}',
                    'recovery_steps': f'{metrics_callback.metrics.get("recovery_steps", 0):.0f}',
                    'action_eff': f'{metrics_callback.metrics.get("action_efficiency", 0):.4f}',
                    'policy_loss': f'{metrics_callback.metrics.get("policy_loss", 0):.4f}',
                    'value_loss': f'{metrics_callback.metrics.get("value_loss", 0):.4f}',
                    'entropy': f'{metrics_callback.metrics.get("entropy", 0):.4f}'
                })
            pbar.update(1)
            
            # Save model very infrequently
            if (i + 1) % (SAVE_EVERY_STEPS * 10) == 0:  # Save much less frequently
                save_path = os.path.join(SAVE_PATH, f"{PPO_MODEL_NAME}_{i+1}")
                model.save(save_path)
        
        pbar.close()
        
        # Save final model
        final_save_path = os.path.join(SAVE_PATH, f"{PPO_MODEL_NAME}_final")
        model.save(final_save_path)
        
        # Save final metrics
        final_metrics_path = os.path.join(SAVE_PATH, f"{PPO_MODEL_NAME}_final_metrics.json")
        metrics_data = {
            'mean_reward': float(metrics_callback.metrics.get('mean_reward', 0)),
            'best_reward': float(metrics_callback.metrics.get('best_reward', 0)),
            'mean_length': float(metrics_callback.metrics.get('mean_length', 0)),
            'load_loss': float(metrics_callback.metrics.get('load_loss', 0)),
            'recovery_steps': int(metrics_callback.metrics.get('recovery_steps', 0)),
            'action_efficiency': float(metrics_callback.metrics.get('action_efficiency', 0)),
            'policy_loss': float(metrics_callback.metrics.get('policy_loss', 0)),
            'value_loss': float(metrics_callback.metrics.get('value_loss', 0)),
            'entropy': float(metrics_callback.metrics.get('entropy', 0)),
            'timestamp': str(time.time()),
            'best_metrics': {k: float(v) for k, v in metrics_callback.best_metrics.items()}
        }
        
        with open(final_metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        return model
        
    finally:
        env.close()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Train the model without tensorboard
    model = train_ppo(use_tensorboard=False)
    print("Training completed successfully!") 