from Agent import Config_PPO
from Agent.PPO import PPO
import warnings

warnings.filterwarnings('ignore')

io_path = 'io/'
ex_path = io_path + 'bank_nnu/'
logs_path = ex_path + 'logs'
session_path = ex_path + 'session'
model_filename = ex_path + 'model'

clustered_devices = None


class bank_nnu:
    def __init__(self, config: Config_PPO):
        self.scope = config.scope
        self.bank = PPO(config=config)  # 生成num个mod

    def choose_action(self, state):
        # --- 【新增】函数，替换旧的 run_enterprise ---
        action = self.bank.choose_action(state)

        return action

    def choose_action_deterministic(self, state):
        action = self.bank.choose_action_deterministic(state)

        return action

    def store_transition(self, state,mu,sigma,action, logprob, reward, is_terminal, next_value, nonterminal):  # CHANGED
        self.bank.store_transition(state,mu,sigma,action, logprob, reward, is_terminal, next_value, nonterminal)

    def get_value(self, state):  # NEW
        return self.bank.get_value(state)

    def learn(self, last_value):
        # --- 【新增】函数 ---
        self.bank.learn(last_value, agent_type=self.scope)

    def clear_memory(self):
        # --- 【新增】函数 ---
        self.bank.clear_memory()

    def log(self):
        # var = self.bank.get_var()
        critic_loss, actor_loss = self.bank.get_loss()
        avg_entropy, avg_clip_frac = self.bank.get_test_indicator()
        return critic_loss, actor_loss, avg_entropy, avg_clip_frac

    def reset_window(self):
        self.state_window.clear()

    # 【新增方法】返回当前历史窗口的快照，用于评估前保存、评估后恢复。
    #  必须可 deepcopy
    def get_window_state(self):
        return list(self.state_window)

    # 恢复历史窗口快照（用于评估结束后继续训练未完成回合）。
    def set_window_state(self, snapshot):
        self.state_window.clear()
        if snapshot is None:
            return
        for s in snapshot:
            self.state_window.append(s)