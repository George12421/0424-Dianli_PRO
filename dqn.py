import os
import json
from tqdm import tqdm
from stable_baselines3 import DQN
from common_utils import create_env, create_gym_env, MetricsCallback, get_common_training_params
from config import *
import torch.nn as nn
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation


class DiscretizeActionWrapper(gym.Wrapper):
    """Wrapper to convert continuous action space to discrete for DQN algorithm.
    
    This wrapper creates a discrete action space by:
    1. Creating a set of predefined actions that cover the most important combinations
    2. Mapping each discrete action to a specific continuous action value
    
    This approach is more computationally efficient than creating a full grid of actions.
    """
    def __init__(self, env, n_actions_per_dim=3, max_actions=729):
        super().__init__(env)
        
        # Get the original continuous action space
        self.original_action_space = env.action_space
        self.low = self.original_action_space.low
        self.high = self.original_action_space.high
        self.action_dims = self.original_action_space.shape[0]
        
        # Determine total discrete actions (limiting to a reasonable number)
        self.n_actions_per_dim = n_actions_per_dim
        
        # Calculate the theoretical total number of actions
        theoretical_actions = n_actions_per_dim ** self.action_dims
        # Limit to a maximum to prevent explosion of action space
        self.n_discrete_actions = min(theoretical_actions, max_actions)
        
        # Create a new discrete action space
        self.action_space = spaces.Discrete(self.n_discrete_actions)
        
        # Generate discretized actions
        self._create_action_map()
        
        print(f"Action space dimensions: {self.action_dims}")
        print(f"Original continuous bounds: {self.low} to {self.high}")
        print(f"Discretized to {self.n_discrete_actions} actions")
    
    def _create_action_map(self):
        """Create a mapping from discrete action indices to continuous action values."""
        self.action_map = {}
        
        if self.n_discrete_actions == self.n_actions_per_dim ** self.action_dims:
            # Full grid of actions if under the limit
            bins = []
            for dim in range(self.action_dims):
                dim_bins = np.linspace(self.low[dim], self.high[dim], self.n_actions_per_dim)
                bins.append(dim_bins)
            
            # Generate all possible combinations
            action_idx = 0
            for indices in self._generate_indices(self.action_dims, self.n_actions_per_dim):
                if action_idx >= self.n_discrete_actions:
                    break
                continuous_action = np.array([bins[dim][idx] for dim, idx in enumerate(indices)])
                self.action_map[action_idx] = continuous_action
                action_idx += 1
        else:
            # If too many combinations, just create a subset of meaningful actions
            # First add actions where one dimension changes at a time
            action_idx = 0
            
            # Add the zero action (all values at middle)
            mid_action = np.mean([self.low, self.high], axis=0)
            self.action_map[action_idx] = mid_action
            action_idx += 1
            
            # Add actions where each dimension is varied separately
            for dim in range(self.action_dims):
                for level in np.linspace(self.low[dim], self.high[dim], self.n_actions_per_dim):
                    if action_idx >= self.n_discrete_actions:
                        break
                    action = mid_action.copy()
                    action[dim] = level
                    self.action_map[action_idx] = action
                    action_idx += 1
            
            # Fill remaining actions with random combinations
            while action_idx < self.n_discrete_actions:
                random_action = np.random.uniform(self.low, self.high)
                self.action_map[action_idx] = random_action
                action_idx += 1
    
    def _generate_indices(self, dims, n_bins):
        """Generate all possible index combinations for the bins."""
        indices = [0] * dims
        while True:
            yield indices.copy()
            
            # Generate next combination
            for i in range(dims - 1, -1, -1):
                indices[i] += 1
                if indices[i] < n_bins:
                    break
                indices[i] = 0
                
                # If we've wrapped around all dimensions, we're done
                if i == 0:
                    return
    
    def step(self, action):
        """Convert discrete action to continuous and apply it."""
        continuous_action = self.action_map[action]
        return self.env.step(continuous_action)


def train_dqn(use_tensorboard=False):
    """Train a DQN agent on the grid2op environment."""
    # Create environment
    env = create_env(ENV_NAME)
    
    try:
        # Create gym environment
        env_gym = create_gym_env(env, OBS_ATTR_TO_KEEP, ACT_ATTR_TO_KEEP)
        
        # Flatten observation space for simpler processing
        env_gym = FlattenObservation(env_gym)
        
        # Discretize the action space for DQN (using fewer bins per dimension)
        # Limit to max 729 discrete actions to keep the action space manageable
        env_gym = DiscretizeActionWrapper(env_gym, n_actions_per_dim=3, max_actions=729)
        
        # Define DQN specific parameters
        dqn_params = {
            "learning_rate": LEARNING_RATE,
            "buffer_size": 10000,  # Replay buffer size
            "learning_starts": 1000,  # How many steps before starting learning
            "batch_size": 32,
            "tau": 0.1,  # Target network update rate
            "gamma": 0.99,  # Discount factor
            "train_freq": 4,  # Update the model every x steps
            "gradient_steps": 1,  # How many gradient steps to do after each rollout
            "target_update_interval": 1000,  # Update the target network every x steps
            "exploration_fraction": 0.1,  # Fraction of entire training period over which exploration rate is reduced
            "exploration_initial_eps": 1.0,  # Initial value of random action probability
            "exploration_final_eps": 0.05,  # Final value of random action probability
            "max_grad_norm": 10,  # Max value for gradient clipping
            "verbose": 0
        }
        
        # Create DQN model
        model = DQN(
            policy="MlpPolicy",
            env=env_gym,
            device="cpu",  # Force CPU usage for better performance
            tensorboard_log=LOGS_DIR if use_tensorboard else None,
            policy_kwargs=dict(
                net_arch=NET_ARCH,
                activation_fn=nn.ReLU
            ),
            **dqn_params
        )
        
        # Create callback for metrics
        metrics_callback = MetricsCallback(model_name=DQN_MODEL_NAME)
        
        # Create progress bar
        pbar = tqdm(total=ITERATIONS, desc="Training DQN", position=0, leave=True)
        
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
                    'q_value': f'{metrics_callback.metrics.get("q_value", 0):.4f}',
                    'loss': f'{metrics_callback.metrics.get("loss", 0):.4f}',
                    'exploration': f'{model.exploration_rate:.4f}'
                })
            pbar.update(1)
            
            # Save model very infrequently
            if (i + 1) % (SAVE_EVERY_STEPS * 10) == 0:  # Save much less frequently
                save_path = os.path.join(SAVE_PATH, f"{DQN_MODEL_NAME}_{i+1}")
                model.save(save_path)
        
        pbar.close()
        
        # Save final model
        final_save_path = os.path.join(SAVE_PATH, f"{DQN_MODEL_NAME}_final")
        model.save(final_save_path)
        
        # Save final metrics
        final_metrics_path = os.path.join(SAVE_PATH, f"{DQN_MODEL_NAME}_final_metrics.json")
        metrics_data = {
            'mean_reward': float(metrics_callback.metrics.get('mean_reward', 0)),
            'best_reward': float(metrics_callback.metrics.get('best_reward', 0)),
            'mean_length': float(metrics_callback.metrics.get('mean_length', 0)),
            'load_loss': float(metrics_callback.metrics.get('load_loss', 0)),
            'recovery_steps': int(metrics_callback.metrics.get('recovery_steps', 0)),
            'action_efficiency': float(metrics_callback.metrics.get('action_efficiency', 0)),
            'q_value': float(metrics_callback.metrics.get('q_value', 0)),
            'loss': float(metrics_callback.metrics.get('loss', 0)),
            'exploration_rate': float(model.exploration_rate),
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
    model = train_dqn(use_tensorboard=False)
    print("Training completed successfully!") 