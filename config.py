# Environment Configuration
ENV_NAME = "l2rpn_case14_sandbox"
# ENV_NAME = "l2rpn_case30"
REWARD_CLASS = "LinesCapacityReward"

# Training Parameters
ITERATIONS = 10_000
LEARNING_RATE = 0.0003
NET_ARCH = [100, 100, 100]
SAVE_EVERY_STEPS = 2000

# Paths
LOGS_DIR = "./logs"
SAVE_PATH = "./saved_model"
PPO_MODEL_NAME = "ppo_grid2op"
PPO2_MODEL_NAME = "ppo_grid2op_PR"
A2C_MODEL_NAME = "a2c_grid2op"
A2C2_MODEL_NAME = "a2c_grid2op_PR"
DQN_MODEL_NAME = "dqn_grid2op"
DQN2_MODEL_NAME = "dqn_grid2op_PR"

# Observation and Action Space
OBS_ATTR_TO_KEEP = [
    "day_of_week", "hour_of_day", "minute_of_hour",
    "prod_p", "prod_v", "load_p", "load_q",
    "actual_dispatch", "target_dispatch", "topo_vect",
    "time_before_cooldown_line", "time_before_cooldown_sub",
    "rho", "timestep_overflow", "line_status",
    "storage_power", "storage_charge"
]

ACT_ATTR_TO_KEEP = [
    "redispatch", "curtail", "set_storage",
]