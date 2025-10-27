__all__ = ['Config']
import random


class MADDPGConfig:
    def __init__(self, scope: str, action_dim: int, action_bound: float, state_dim: int = 0, reward_gamma: int = 0.95, memory_capacity: int = 80000,
                 learning_rate_actor: float = 0.0001, learning_rate_critic: float = 0.0002,
                 learning_rate_actor_stable: float = 0.0001,learning_rate_critic_stable:float = 0.0002,
                 learning_rate_decay: float = 0.98,learning_rate_decay_time:int=200,
                 soft_replace_tau: float = 0.01, batch_size: int = 30000, var_init: float = 0.5, var_stable: float = 0.001,
                 var_drop_at: int = 3000, var_stable_at: int = 50000, var_end_at=float('inf'),random_seed:int = None,
                 smooth_noise: float = 0.01, actor_update_delay_times: int = 3,
                 is_critic_double_network: bool = True,is_actor_update_delay:bool = True,is_QNet_smooth_critic:bool = True):
        self.scope = scope
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.REWARD_GAMMA = reward_gamma
        self.MEMORY_CAPACITY = memory_capacity
        self.LEARNING_RATE_ACTOR = learning_rate_actor
        self.LEARNING_RATE_CRITIC = learning_rate_critic
        self.LEARNING_RATE_ACTOR_STABLE = learning_rate_actor_stable
        self.LEARNING_RATE_CRITIC_STABLE = learning_rate_critic_stable
        self.LEARNING_RATE_DECAY=learning_rate_decay
        self.LEARNING_RATE_DECAY_TIME=learning_rate_decay_time
        self.SOFT_REPLACE_TAU = soft_replace_tau
        self.BATCH_SIZE = batch_size
        self.VAR_INIT = var_init
        self.VAR_STABLE = var_stable
        self.VAR_DROP_AT = var_drop_at
        self.VAR_STABLE_AT = var_stable_at
        self.VAR_END_AT = var_end_at
        self.SMOOTH_NOISE = smooth_noise
        self.ACTOR_UPDATE_DELAY_TIMES = actor_update_delay_times
        self.IS_ACTOR_UPDATE_DELAY = is_actor_update_delay
        self.IS_CRITIC_DOUBLE_NETWORK = is_critic_double_network
        self.IS_QNET_SMOOTH_CRITIC = is_QNet_smooth_critic
        if random_seed is None:
            self.random_seed = random.randint(1,1000)
        else:
            self.random_seed = random_seed

    def set_state_dim(self, state_num: int):
        self.state_dim = state_num

    def set_scope(self, scope: str):
        self.scope = scope

    def set_learning_rate(self, learning_rate_actor:float = None, learning_rate_critic:float = None ):
        if learning_rate_actor is not None:
            self.LEARNING_RATE_ACTOR = learning_rate_actor
        if learning_rate_critic is not None:
            self.LEARNING_RATE_CRITIC = learning_rate_critic
    def set_seed(self,seed):
        self.random_seed = seed

    def __str__(self):
        res = ''
        for key in self.__dict__:
            print(self.__dict__[key])
            res = res + key + ' : ' + str(self.__dict__[key]) + '\n'
        return res


