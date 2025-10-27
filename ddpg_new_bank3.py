'''
ep = 企业, mod = 企业家, 一个episode = 一个企业从生到死
bank = bank_nnu(num:int)//传入企业家的个数,生成num个mod
bank.run_bank(bank_mod, state)//传入经营的企业家编号&企业状态,输出[银行的贷款额度, 银行的利息]
'''

import warnings
from Cortex import *
from Agent.DDPG import DDPG
# from Cortex.ActorDQN import *
# from Cortex.Common.Network import leaky_relu
# from Cortex.Common.Network import AdamOptimizer
# from Cortex.Common.Network import TF_Neural_Network as Network
# from Cortex.Common.ExperienceReplay import pick_selector_class as PickSelectorClass
import warnings
import time
import numpy as np
import copy
import pandas as pd
from new_calculate import *

warnings.filterwarnings('ignore')

io_path = 'io/'
ex_path = io_path + 'bank_nnu/'
logs_path = ex_path + 'logs'
session_path = ex_path + 'session'
model_filename = ex_path + 'model'

clustered_devices = None


def _custom_reward(state, day):
    reward = []
    for i in range(len(state)):
        r = state[i][0]   # M Ω0 Ω1 D0 D1 WNDB0 WNDB1 real_WNDB0 real_WNDB1 WNDF0 WNDF1←下标10 π0 π1  X0 X1 iD0 iD1 NP0 NP1
        if state[i][5] + state[i][6] > state[i][7] + state[i][8]:
            r = state[i][7] + state[i][8] - state[i][5] - state[i][6]
        # r = day
        # r = state[i][E_DATA.π.value]
        # r = state[i][1] + 100*state[i][7]
        # 修改点
        # if new_ep[i]: r = -1000
        r /= 100
        reward.append(r)
    print('reward',reward)
    return reward



experience_size = 50000   # 经验库的大小
train_seq_len = 1  # 样本的时间序列长度
allow_short_seq = False  # 是否允许取序列长度不足seq len所指定的长度
train_batch_size = 10000  # 一次取多少样本出来训练 LSTM 10 else 200
valid_sample_rate = 0.1  # 当经验库中样本达到 batch size * rate时，反复抽取经验库中的经验进行训练
feeding_buffer = 0  # 用于异步进行计算q target，如果这个值不是0，
                    # 则会建立专门线程，将抽取的batch先计算q-target，
                    # train的时候直接使用，这个值你可以理解为预先计算多少个batch的q-target

ddpg_state_size = 19  # 神经网络的输入数据state有8个dimension
ddpg_action_num = 2  # action的取值范围[0,action_num - 1]

q_decay = 0.95
upd_shadow_period = 100  # 训练多少次后更新 q_target

learning_rate = 0.01
dropout_rate = 0.02

is_random = True
is_max = True

class bank_nnu:
    def __init__(self, num: int,scope):
        self.bank = DDPG(ddpg_action_num,ddpg_state_size,0.49,scope)  # 生成num个mod
        self.epi_map = [None] * num  # 把mod和ep形成映射 epi_map = [mod0_ep, mod1_ep, mod2_ep] mod当前运营的ep在h_epi中的编号
        self.epi_num = [0] * num  # 银行家i运营银行数
        self.epi_step = [0] * num  # mod的ep的决策步数，epi_step = [mod0_ep_step,mod1_ep_step,...,mod[num]_ep_step]
        self.epi_day = [0] * num  # epi_day记录银行存活天数
        self.last_state = [None] * num  # 银行家i的银行的上一个state
        # self.bank.print_graphs_thread(logs_path, True)  # 生成神经网络的初始结构图，保存在io/bank_nnu/logs，用tensorboard查看

    def run_bank(self, mod, state):  # bank_mod范围:[1, num]
        if is_random:
            if is_max:
                return [np.array([[0.5,0.5]])]
            else:
                return [np.array([[random.random()-0.5,random.random()-0.5]])]
        mod_num = len(mod)
        # =====准备工作=====#
        h_epi = [None] * mod_num  # 准备h_epi
        new_ep = [True] * mod_num  # 是否是新的ep
        state = list(state)
        for i in range(mod_num):
            state[i] = np.array(state[i])  # state准备就绪
            if mod[i] < 0:
                mod[i] = (-mod[i]) - 1
            else:
                new_ep[i] = False

        # =====得到action=====#
        for i in range(mod_num):
            if not new_ep[i]:
                h_epi[i] = self.epi_map[mod[i]]
        h_epi, action = self.bank.choose_action(h_epi, state)
        # =====更新self.epi_map self.epi_step,并对action进行处理=====#
        for i in range(mod_num):
            if new_ep[i]:
                self.epi_num[mod[i]] += 1
                self.epi_day[mod[i]] = 1
                self.epi_step[mod[i]] = 1
                self.epi_map[mod[i]] = copy.deepcopy(h_epi[i])
            else:  # 旧银行
                self.epi_step[mod[i]] += 1

        return action

    def env_upd(self, mod, state, day, is_train,is_End = False):  #  这里的h_epi是upd_epi
        mod_num = len(mod)
        if is_random:
            return 0
        # =====准备工作=====#
        upd_mod = [];
        upd_state = [];
        upd_epi = [];
        h_epi = [None] * mod_num  # 准备h_epi
        new_ep = [True] * mod_num  # 是否是新的ep
        state = list(state)
        for i in range(mod_num):
            state[i] = np.array(state[i])  # state准备就绪
            upd_mod.append(mod[i])
            upd_epi.append(self.epi_map[mod[i]])
            upd_state.append(state[i])
            self.last_state[mod[i]] = state[i]
        print(self.epi_map)
        # =====得到reward=====#
        reward = _custom_reward(state, day)
        # =====把[state, action, reward, next_state]按序存入h_epi所属部分, 更新epi_map=====#
        for i in range(mod_num):
            # flag[i] += reward[i]

            h_epi[i] = self.bank.episode_feedback(upd_epi[i], reward[i], upd_state[i] if is_End else None)
            self.epi_map[mod[i]] = h_epi[i]
        if is_train:
            loss = self.bank.learn()
            return loss[0]

    def bank_mod_close(self, bank_mod):  # bank_mod可以是int或list
        mod = bank_mod
        if type(mod) == int:
            mod = [mod]
        mod = np.array(mod)
        mod_num = len(mod)
        state = [None] * mod_num
        h_epi = [None] * mod_num
        new_ep = [True] * mod_num
        for i in range(mod_num):
            mod[i] -= 1
            state[i] = self.last_state[mod[i]]
            h_epi[i] = self.epi_map[mod[i]]
        self.env_upd(mod, state, h_epi, new_ep)
        self.bank.close()
        print('Close all bank_mod')

class bank_nnu_without_intelligent:
    def __init__(self, num: int, name):
        self.num = num
        self.name = "请输入"+name+"的动作决策,顺序为 WNDB1 WNDB2: "
        # self.input = pd.read_csv('C://Users//30800//Desktop//input.csv')

    def run_bank(self, mod, state,episode):
        # action应为[ WNDB1,WNDB2 ]
        # WNDB1 = self.input['WNDB1'][episode]
        # WNDB2 = self.input['WNDB2'][episode]
        # print(self.name+str([WNDB1,WNDB2]))

        # WNDB1,  WNDB2= map(float, input(self.name).split())
        # return [WNDB1, WNDB2]
        return [100, 100]

