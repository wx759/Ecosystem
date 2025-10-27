import random


class Config_PPO:
    def __init__(self,
                 scope: str,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 # 基础参数
                 gamma: float = 0.99,
                 lamda: float = 0.95,

                 learning_rate_actor_bank: float = 0.00007,
                 learning_rate_actor_enterprise: float = 0.00008,
                 learning_rate_critic_bank: float = 0.0008,
                 learning_rate_critic_enterprise: float = 9e-5,
                 mini_batch_size: int = 128,

                 # PPO核心参数
                 update_timestep: int = 2048,
                 max_training_steps: int = 2000000,
                 total_step: int = 3000000,

                 clip_range: float = 0.1,
                 n_epochs: int = 8,

                 entropy_enterprise: float = 0.0065,
                 entropy_bank: float = 0.0035,

                 # 设置参数
                 is_rms_state: bool = True,
                 is_rms_reward: bool = True,

                 # 随机种子
                 random_seed: int = 105,
                 ):
        self.total_step = total_step
        self.entropyRC_Bank = entropy_bank
        self.entropyRC_Enterprise = entropy_enterprise
        self.scope = scope
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        # 基础参数
        self.GAMMA = gamma
        self.LAMDA = lamda

        self.LEARNING_RATE_AC_Bank = learning_rate_actor_bank
        self.LEARNING_RATE_AC_Enterprise = learning_rate_actor_enterprise
        self.LEARNING_RATE_C_Bank = learning_rate_critic_bank
        self.LEARNING_RATE_C_Enterprise = learning_rate_critic_enterprise
        self.MINI_BATCH_SIZE = mini_batch_size

        # PPO核心参数
        self.CLIP_RANGE = clip_range
        self.N_EPOCHS = n_epochs
        self.UPDATE_TIMESTEP = update_timestep
        self.MAX_TRAINING_STEPS = max_training_steps
        # self.entropyRC = entropyRC
        self.total_update = max_training_steps / update_timestep
        # 随机种子
        if random_seed is None:
            self.random_seed = random.randint(1, 1000)  # 给一个固定默认值，或者 None
        else:
            self.random_seed = random_seed
        self.is_rms_state = is_rms_state
        self.is_rms_reward = is_rms_reward

    def set_state_dim(self, state_dim: int):
        self.state_dim = state_dim

    def set_scope(self, scope: str):
        self.scope = scope

    def __str__(self):
        return "\n".join([f"{k: <20}: {v}" for k, v in self.__dict__.items()])
