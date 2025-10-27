'''
ep = 企业, mod = 企业家, 一个episode = 一个企业从生到死
bank = bank_nnu(num:int)//传入企业家的个数,生成num个mod
bank.run_bank(bank_mod, state)//传入经营的企业家编号&企业状态,输出[银行的贷款额度, 银行的利息]
'''

from Agent.DDPG import DDPG
from Agent.Config import Config

import warnings
import copy

from new_calculate import *

warnings.filterwarnings('ignore')

io_path = 'io/'
ex_path = io_path + 'bank_nnu/'
logs_path = ex_path + 'logs'
session_path = ex_path + 'session'
model_filename = ex_path + 'model'

clustered_devices = None



class bank_nnu:
    def __init__(self, config: Config):
        self.scope = config.scope
        self.bank = DDPG(config=config)  # 生成num个mod
        self.epi = None
        self.last_state = None  # 银行家i的银行的上一个state

    def run_bank(self, state, new_ep):  # bank_mod范围:[1, num]
        # =====准备工作=====#
        if new_ep:
            h_epi = None  # 准备h_epi
        else:
            h_epi = self.epi
        state = np.array(state)  # state准备就绪

        # =====得到action=====#
        h_epi, action = self.bank.choose_action(h_epi, state)
        if new_ep:
            self.epi = copy.deepcopy(h_epi)
        for i in range(len(action)):
            action[i] = action[i] + 0.5 # 银行动作为0~1

        return action

    def env_upd(self, state, action, state_, reward, is_train, is_End = False):  #  这里的h_epi是upd_epi

        # =====准备工作=====#
        state_ = np.array(state_)  # state准备就绪
        self.last_state = state_
        # print(self.scope, ":", self.epi)
        # =====把[state, action, reward, next_state]按序存入h_epi所属部分, 更新epi_map=====#

        if is_train:
            self.epi = self.bank.episode_feedback(self.epi, state, action, reward, state_ if is_End else None)
            loss = self.bank.learn()
            return loss
        return None

    def mark(self):
        return self.bank.mark()

