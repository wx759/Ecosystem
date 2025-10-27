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
from Agent.TD3 import TD3
from Agent.PPO import PPO
from Agent.RuningMeanStd import RunningMeanStd

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
        self.scope = config.scope
        self.enterprise = PPO(config=config)  # 生成num个mod
        self.rms = RunningMeanStd(shape=33)
        # 添加一个控制归一化开关的标志，方便你测试对比效果
        self.is_rms = True  # 初始设置为 True，如果你想暂时关闭可以改为 False
        self.device = "cpu"
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

    def choose_action(self, state):
        # --- 【新增】函数，替换旧的 run_enterprise ---
        return self.enterprise.choose_action(state)

    def choose_action_deterministic(self, state):
        action = self.enterprise.choose_action_deterministic(state)

        return action

    def store_transition(self, state, action, logprob, reward, is_terminal, next_value, nonterminal):  # CHANGED
        self.enterprise.store_transition(state, action, logprob, reward, is_terminal, next_value, nonterminal)

    def get_value(self, state):  # NEW
        return self.enterprise.get_value(state)

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
