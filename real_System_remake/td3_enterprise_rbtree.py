'''
ep = 企业, mod = 企业家, 一个episode = 一个企业从生到死
enterprise = enterprise_nnu(num:int)//传入企业家的个数,生成num个mod
enterprise._run_enterpeise(enterprise_mod:int, state:list)//传入经营的企业家编号&企业状态,
'''

import warnings

from Agent import Config
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
from Agent.TD3_rb_tree import TD3
# from Agent.TD3withoutNoise import TD3
warnings.filterwarnings('ignore')

io_path = 'io/'
ex_path = io_path + 'enterprise_nnu/'
logs_path = ex_path + 'logs'
session_path = ex_path + 'session'
model_filename = ex_path + 'model'

clustered_devices = None





class enterprise_nnu:
    def __init__(self, config: Config):
        self.scope = config.scope
        self.enterprise = TD3(config=config)  # 生成num个mod
        self.epi = None
        self.last_state = None  # 银行家i的银行的上一个state
            
    def run_enterprise(self, state, new_ep):  # enterprise_mod范围:[1, num]
        # =====准备工作=====#
        if new_ep:
            h_epi = None  # 准备h_epi
        else:
            h_epi = self.epi
        state = np.array(state)  # state准备就绪

        # =====得到action=====#
        h_epi, action = self.enterprise.choose_action(h_epi, state)
        if new_ep:
            self.epi = copy.deepcopy(h_epi)

        return action

    def env_upd_rbtree(self, state, action, state_, reward):
        # =====翻转=======#
        # 只需要转状态
        # state
        temp_s=state[13:21]
        state_r = state
        # 13 14 <- 23 24,15 16 <- 21 22,17 18 19 20 <- 25 26 27 28
        state_r[13:15] = state[23:25]
        state_r[15:17] = state[21:23]
        state_r[17:21] = state[25:29]
        state_r[23:25] = temp_s[13:15]
        state_r[21:23] = temp_s[15:17]
        state_r[25:29] = temp_s[17:21]
        # next state
        temp_s_ = state_[13:21]
        state_r_ = state_
            # 13 14 <- 23 24,15 16 <- 21 22,17 18 19 20 <- 25 26 27 28
        state_r_[13:15] = state[23:25]
        state_r_[15:17] = state[21:23]
        state_r_[17:21] = state[25:29]
        state_r_[23:25] = temp_s_[13:15]
        state_r_[21:23] = temp_s_[15:17]
        state_r_[25:29] = temp_s_[17:21]

        # =====把[state, action, reward, next_state]按序存入h_epi所属部分=====#
        self.enterprise.store_transition_rbtree(self.epi, state_r,action, reward, state_r_)


    def env_upd(self, state, action, state_, reward, is_train, is_end=False):  # 这里的h_epi是upd_epi
        # =====准备工作=====#
        state_ = np.array(state_)  # state准备就绪
        self.last_state = state_
        # print(self.scope, ":", self.epi)
        # =====把[state, action, reward, next_state]按序存入h_epi所属部分, 更新epi_map=====#

        if is_train:
            self.epi = self.enterprise.episode_feedback(self.epi, state, action, reward, state_ if is_end else None)
            loss = self.enterprise.learn()
            return loss
        return None

    def get_show(self):
        return self.enterprise.check_show()



