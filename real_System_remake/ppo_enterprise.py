'''
ep = 企业, mod = 企业家, 一个episode = 一个企业从生到死
enterprise = enterprise_nnu(num:int)//传入企业家的个数,生成num个mod
enterprise._run_enterpeise(enterprise_mod:int, state:list)//传入经营的企业家编号&企业状态,
'''

import warnings

import torch

from Agent import Config_PPO
from Cortex import *
# from Cortex.ActorDQN import *
# from Cortex.Common.Network import leaky_relu
# from Cortex.Common.Network import AdamOptimizer
# from Cortex.Common.Network import TF_Neural_Network as Network
# from Cortex.Common.ExperienceReplay import pick_selector_class as PickSelectorClass
import warnings
import copy
import time
import numpy as np
import pandas as pd
# import tensorflow as tf
from new_calculate import *
# from Agent.DDPG import DDPG
# from Agent.TD3 import TD3
from Agent.PPO import PPO
from Agent.RuningMeanStd import RunningMeanStd
from collections import deque
# from Agent.TD3_attention import TD3 as TD3_attn
# from Agent.TD3withoutNoise import TD3
warnings.filterwarnings('ignore')

io_path = 'io/'
ex_path = io_path + 'enterprise_nnu/'
logs_path = ex_path + 'logs'
session_path = ex_path + 'session'
model_filename = ex_path + 'model'

clustered_devices = None


class enterprise_nnu:
    def __init__(self, config: Config_PPO):
        self.current_seq_state = None
        self.scope = config.scope
        self.enterprise = PPO(config=config)  # 生成num个mod
        self.rms = RunningMeanStd(shape=33)
        # 添加一个控制归一化开关的标志，方便你测试对比效果
        self.is_rms = True  # 初始设置为 True，如果你想暂时关闭可以改为 False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # === 新增：历史状态队列 ===
        self.seq_len = config.seq_len
        self.state_window = deque(maxlen=self.seq_len)
        # # 初始化填满0，防止一开始空报错
        # for _ in range(self.seq_len):
        #     self.state_window.append(np.zeros(config.state_dim))

        # =======================
        # self.epi = None
        # self.last_state = None  # 银行家i的银行的上一个state

    # def run_enterprise(self, state, new_ep):  # enterprise_mod范围:[1, num]
    #     # =====准备工作=====#
    #     if new_ep:
    #         h_epi = None  # 准备h_epi
    #     else:
    #         h_epi = self.epi
    #     state = np.array(state)  # state准备就绪
    #
    #     # =====得到action=====#
    #     # h_epi, action = self.enterprise.choose_action_attn(h_epi, state)
    #     h_epi, action = self.enterprise.choose_action(h_epi, state)  # 银行和企业分开的原因 企业动作输出范围是-0.5-0.5 银行动作输出范围是0-1
    #     if new_ep:
    #         self.epi = copy.deepcopy(h_epi)
    #
    #     return action

    def choose_action(self, raw_state):
        # --- 【新增】函数，替换旧的 run_enterprise ---
        # return self.enterprise.choose_action(state)
        # --- 【修改逻辑】 ---

        # 如果窗口是空的（刚开始），或者之前的都是初始化的0（如果保留原逻辑）
        # 建议直接判断 len 或增加一个 reset 标志
        if len(self.state_window) == 0:
            # 冷启动：用当前第一帧状态填满整个历史窗口
            for _ in range(self.seq_len):
                self.state_window.append(raw_state)
        else:
            # 正常步进
            self.state_window.append(raw_state)

        # 2. 制作 Transformer 需要的 "State"
        # shape: (seq_len, 35)
        seq_state = np.array(self.state_window)

        # 3. 传给 Actor 选择动作
        # 注意：这里要把 seq_state 存下来！不仅仅是 raw_state
        self.current_seq_state = seq_state

        return self.enterprise.choose_action(seq_state)

    def choose_action_deterministic(self, state):

        if len(self.state_window) == 0:
            for _ in range(self.seq_len):
                self.state_window.append(state)
        else:
            self.state_window.append(state)

        # 2. 制作 Transformer 需要的 "State"
        seq_state = np.array(self.state_window)
        action = self.enterprise.choose_action_deterministic(seq_state)

        return action

    def store_transition(self, state, mu,sigma,action, logprob, reward, is_terminal, next_value, nonterminal):  # CHANGED
        self.enterprise.store_transition(self.current_seq_state, mu,sigma,action, logprob, reward, is_terminal, next_value, nonterminal)

    def get_value(self, state):  # NEW
        return self.enterprise.get_value(state)

    def get_next_value(self, raw_next_state):
        """
        计算 V(s_{t+1})（严格版）：
        - 要求当前 state_window 非空（否则说明调用顺序不对）
        - 使用 state_window 的快照 + raw_next_state 构造 next_seq_state（不污染真实窗口）
        - 把 next_seq_state 交给 PPO.get_value（其内部会加 batch 维 -> (1,S,D)）
        """
        assert len(self.state_window) > 0, (
            f"[{self.scope}] state_window is empty when computing next_value; "
            f"call choose_action() first."
        )

        tmp = deque(self.state_window, maxlen=self.seq_len)
        tmp.append(raw_next_state)  # 推进到 t+1

        next_seq_state = np.array(tmp, dtype=np.float64)  # 与你 PPO.get_value dtype 习惯一致
        return self.enterprise.get_value(next_seq_state)

    def learn(self, last_value):
        # --- 【新增】函数 ---
        self.enterprise.learn(last_value, agent_type=self.scope)

    def clear_memory(self):
        # --- 【新增】函数 ---
        self.enterprise.clear_memory()

    def log(self):
        # var = self.enterprise.get_var()
        critic_loss, actor_loss = self.enterprise.get_loss()
        avg_entropy, avg_clip_fraction = self.enterprise.get_test_indicator()
        return critic_loss, actor_loss, avg_entropy, avg_clip_fraction

    # def get_show(self):
    #     return self.enterprise.check_show()

    # 【新增方法】专门用于回合结束时清空历史
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