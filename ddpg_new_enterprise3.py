'''
ep = 企业, mod = 企业家, 一个episode = 一个企业从生到死
enterprise = enterprise_nnu(num:int)//传入企业家的个数,生成num个mod
enterprise._run_enterpeise(enterprise_mod:int, state:list)//传入经营的企业家编号&企业状态,
'''

import warnings
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
from new_calculate import *
from Agent.DDPG import DDPG

warnings.filterwarnings('ignore')

io_path = 'io/'
ex_path = io_path + 'enterprise_nnu/'
logs_path = ex_path + 'logs'
session_path = ex_path + 'session'
model_filename = ex_path + 'model'

clustered_devices = None

def _custom_reward(state, is_Fall, day):
    reward = []
    for i in range(len(state)):
        r = state[i][7]   # M X D R iD C π total_π P K L ←下标10 WNDF NP getK getL k l
        if is_Fall:
            r = -500
        else:
            if state[i][1] == 0.0:
                r =- 1000
        # if day == 0:
        #     r = -100
        # r = day
        # r = state[i][E_DATA.π.value]
        # r = state[i][1] + 100*state[i][7]
        # 修改点
        # if new_ep[i]: r = -1000
        r /= 100
        reward.append(r)
    print('reward',reward)
    return reward


experience_size = 80000  # 经验库的大小
train_seq_len = 1  # 样本的时间序列长度
allow_short_seq = False  # 是否允许取序列长度不足seq len所指定的长度
train_batch_size = 10000  # 一次取多少样本出来训练 用LSTM后和银行都要改成10 else 300
valid_sample_rate = 0.1  # 当经验库中样本达到 batch size * rate时，反复抽取经验库中的经验进行训练
feeding_buffer = 0  # 用于异步进行计算q target，如果这个值不是0，
                    #   则会建立专门线程，将抽取的batch先计算q-target，train的时候直接使用，
                    #   这个值你可以理解为预先计算多少个batch的q-target

ddpg_state_size = 17  # 神经网络的输入数据state有state_size个dimension
ddpg_action_num = 4   # action的取值范围[0,action_num - 1]

q_decay = 0.95
upd_shadow_period = 100  # 训练多少次后更新 q_target

learning_rate = 0.01
dropout_rate = 0.02

isRandom = True
isMax = True


class enterprise_nnu:
    def __init__(self, num: int,scope):
        self.enterprise = DDPG(ddpg_action_num,ddpg_state_size,0.49,scope)   # 生成num个mod
        self.epi_map = [None] * num
        self.epi_num = [0] * num
        self.epi_step = [0] * num
        self.epi_day = [0] * num  # epi_day记录企业存活天数
        self.last_state = [None] * num
        # self.enterprise.print_graphs_thread(logs_path, True)
            
    def run_enterprise(self, mod, state):  # enterprise_mod范围:[1, num]
        if isRandom:
            if isMax:
                return np.array([[0.5,0.5,0.5,0.5]])
            else:
                return np.array([[random.random()-0.5,random.random()-0.5,random.random()-0.5,random.random()-0.5]])
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
        h_epi, action = self.enterprise.choose_action(h_epi, state)
        # =====更新self.epi_map self.epi_step,并对action进行处理=====#
        for i in range(mod_num):
            if new_ep[i]:
                self.epi_num[mod[i]] += 1
                self.epi_day[mod[i]] = 1
                self.epi_step[mod[i]] = 1
                self.epi_map[mod[i]] = copy.deepcopy(h_epi[i])
            else:  # 旧企业
                self.epi_step[mod[i]] += 1

        return action

    def env_upd(self, mod, state, day, is_train,is_End = False,is_Fall = False):  # bank_mod范围:[1, num]
        mod_num = len(mod)
        if isRandom:
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
        reward = _custom_reward(state, is_Fall,day)
        # =====把[state, action, reward, next_state]按序存入h_epi所属部分, 更新epi_map=====#
        for i in range(mod_num):
            # flag[i] += reward[i]

            h_epi[i] = self.enterprise.episode_feedback(upd_epi[i], reward[i], upd_state[i] if is_End else None)
            self.epi_map[mod[i]] = h_epi[i]

        if is_train:
            loss = self.enterprise.learn()
            print("loss",loss)
            return loss[0]
        return None


                      
    def enterprise_mod_close(self, enterprise_mod):  # bank_mod可以是int或list
        mod = enterprise_mod
        if type(mod) == int:
            mod = [mod]
        mod = np.array(mod)
        mod_num = len(mod)
        state = [None] * mod_num
        h_epi = [None] * mod_num
        new_ep = [True] * mod_num
        for i in range(mod_num):
            mod[i] -= 0
            state[i] = self.last_state[mod[i]]
            h_epi[i] = self.epi_map[mod[i]]
        self.env_upd(mod, state, h_epi, new_ep, False)
        self.enterprise.close()
        print('Close all enterprise_mod')


class enterprise_nnu_without_intelligent:
    def __init__(self, num: int, name):
        self.num = num
        self.name = "请输入"+name+"的动作决策,顺序为 WNDF K L NP: "
        self.input = pd.read_csv('C://Users//30800//Desktop//input.csv')

    def run_enterprise(self, mod, state, episode, show=False):
        # action应为[WNDF, K, L, NP]
        no = mod[0] + 1
        # WNDF, K, L, NP = map(float, input(self.name).split())
        WNDF = self.input['WNDF'+str(no)][episode]
        K = self.input['K'+str(no)][episode]
        L = self.input['L'+str(no)][episode]
        NP = self.input['NP'+str(no)][episode]
        print(self.name+str([WNDF, K, L, NP]))
        if show:
            input()
        return [WNDF, K, L, NP]


