'''
ep = 企业, mod = 企业家, 一个episode = 一个企业从生到死
enterprise = enterprise_nnu(num:int)//传入企业家的个数,生成num个mod
enterprise._run_enterpeise(enterprise_mod:int, state:list)//传入经营的企业家编号&企业状态,
'''

import warnings

from . import MADDPGConfig
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
import tensorflow as tf
from real_System.MADDPG.MADDPG import MADDPG

warnings.filterwarnings('ignore')

io_path = 'io/'
ex_path = io_path + 'enterprise_nnu/'
logs_path = ex_path + 'logs'
session_path = ex_path + 'session'
model_filename = ex_path + 'model'

clustered_devices = None





class enterprise_nnu:
    def __init__(self, config: MADDPGConfig):
        self.scope = config.scope
        self.enterprise = MADDPG(config=config)  # 生成num个mod
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

    def env_upd(self, state,other_state, action,other_action, state_,other_state_,
                reward,  is_end=False):  # 这里的h_epi是upd_epi
        # =====准备工作=====#
        state_ = np.array(state_)  # state准备就绪
        self.last_state = state_
        # print(self.scope, ":", self.epi)
        # =====把[state, action, reward, next_state]按序存入h_epi所属部分, 更新epi_map=====#

        self.epi = self.enterprise.episode_feedback(self.epi, state,other_state,
                                                    action,other_action,
                                                    reward,
                                                    state_ if is_end else None,other_state_ if is_end else None)


    def get_batch(self):
        return self.enterprise.get_batch()

    def get_other_agent_policy_action(self,batch):
        return self.enterprise.get_action_to_enviroment(batch)

    def learn(self,action_feedback,action__feedback,is_train,):
        if is_train:
            loss = self.enterprise.learn(action_feedback,action__feedback)
            return loss
        return None

    def get_show(self):
        return self.enterprise.check_show()



