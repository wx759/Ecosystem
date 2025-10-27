__all__ = ['Config']


DEFAULT_Q_DELAY = 0.95
DEFAULT_UPD_SHADOW_PERIOD = 1000


class Config:
    def __init__(self, experience_size: int = 0, pick_len: int = 1, allow_short_seq: bool = False,
                 train_batch_size: int = 1, valid_sample_rate: float = 0, Q_network_func=None,
                 R_network_func=None, s_shape=None, q_decay: float = DEFAULT_Q_DELAY,
                 upd_shadow_period: int = DEFAULT_UPD_SHADOW_PERIOD, optimizer=None,
                 double_dqn: bool = False, pick_selector_class=None):
        self.experience_size = experience_size
        self.pick_len = pick_len
        self.allow_short_seq = allow_short_seq
        self.train_batch_size = train_batch_size
        self.valid_sample_rate = valid_sample_rate
        self.Q_network_func = Q_network_func
        self.R_network_func = R_network_func
        self.s_shape = s_shape
        self.q_decay = q_decay
        self.upd_shadow_period = upd_shadow_period
        self.optimizer = optimizer
        self.double_dqn = double_dqn
        self.pick_selector_class = pick_selector_class
        self.act_num = 0
