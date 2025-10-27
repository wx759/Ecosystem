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
import time
import numpy as np
import pandas as pd
from new_calculate import *

warnings.filterwarnings('ignore')

io_path = 'io/'
ex_path = io_path + 'enterprise_nnu/'
logs_path = ex_path + 'logs'
session_path = ex_path + 'session'
model_filename = ex_path + 'model'

clustered_devices = None

def _custom_reward(state, new_ep, day):
    reward = []
    for i in range(len(state)):
        r = state[i][6]   # M X D R iD C π total_π P
        # r = day
        # r = state[i][E_DATA.π.value]
        # r = state[i][1] + 100*state[i][7]
        # 修改点
        # if new_ep[i]: r = -1000
        reward.append(r)
    # print('reward',reward)
    return reward


experience_size = 80000  # 经验库的大小
train_seq_len = 1  # 样本的时间序列长度
allow_short_seq = False  # 是否允许取序列长度不足seq len所指定的长度
train_batch_size = 300  # 一次取多少样本出来训练 用LSTM后和银行都要改成10 else 300
valid_sample_rate = 0.1  # 当经验库中样本达到 batch size * rate时，反复抽取经验库中的经验进行训练
feeding_buffer = 0  # 用于异步进行计算q target，如果这个值不是0，
                    #   则会建立专门线程，将抽取的batch先计算q-target，train的时候直接使用，
                    #   这个值你可以理解为预先计算多少个batch的q-target

state_size = 9  # 神经网络的输入数据state有state_size个dimension
action_num = 10000  # action的取值范围[0,action_num - 1]

q_decay = 0.95
upd_shadow_period = 100  # 训练多少次后更新 q_target

learning_rate = 0.01
dropout_rate = 0.00

#noise_net 版本
# def Q_network_func(network:Network, state_name:str, q_name:str, has_shadow:bool):
#     network.add_layer_noisy('FC1', state_name, 'H1', state_size * 8, act_func=ActivationFunc.leaky_relu,
#                             dropout_rate_name='dropout_rate', has_shadow=has_shadow)
#     network.add_layer_full_conn('FC2', 'H1', 'H2', state_size * 8, act_func=ActivationFunc.leaky_relu,
#                                 dropout_rate_name='dropout_rate', has_shadow=has_shadow)
#     network.add_layer_full_conn('FC3', 'H2', 'H3', state_size * 4, act_func=ActivationFunc.leaky_relu,
#                                 dropout_rate_name='dropout_rate', has_shadow=has_shadow)
#     network.add_layer_noisy('FC4', 'H3', 'A', action_num, has_shadow=has_shadow)
#     network.add_layer_noisy('FC5', 'H3', 'V', 1, has_shadow=has_shadow)
#     network.add_layer_duel_q('Duel_Q', 'V', 'A', q_name, has_shadow=has_shadow)

# lstm版本
# def Q_network_func(network: Network, state_name: str, q_name: str, has_shadow: bool):
#     network.add_layer_lstm('LSTM1', state_name, 'H1', state_size * 8, has_shadow=has_shadow)
#     network.add_layer_batch_norm('BN1', 'H1', 'H1_norm')
#     network.add_layer_full_conn('FC1', 'H1_norm', 'H3', state_size * 8, act_func=ActivationFunc.leaky_relu,
#                                 dropout_rate_name='dropout_rate', batch_norm=True, has_shadow=has_shadow)
#     network.add_layer_full_conn('FC3', 'H3', 'H4', state_size * 4, act_func=ActivationFunc.leaky_relu,
#                                 dropout_rate_name='dropout_rate', batch_norm=True, has_shadow=has_shadow)
#     network.add_layer_full_conn('FC4', 'H4', 'A', action_num, has_shadow=has_shadow)
#     network.add_layer_full_conn('FC5', 'H4', 'V', 1, has_shadow=has_shadow)
#     network.add_layer_duel_q('Duel_Q', 'V', 'A', q_name, has_shadow=has_shadow)

#原始版本
def Q_network_func(network:Network, state_name:str, q_name:str, has_shadow:bool):
    network.add_layer_full_conn('FC1', state_name, 'H1', state_size * 8, act_func=ActivationFunc.leaky_relu,
                                dropout_rate_name='dropout_rate', has_shadow=has_shadow)
    network.add_layer_full_conn('FC2', 'H1', 'H2', state_size * 8, act_func=ActivationFunc.leaky_relu,
                                dropout_rate_name='dropout_rate', has_shadow=has_shadow)
    network.add_layer_full_conn('FC3', 'H2', 'H3', state_size * 4, act_func=ActivationFunc.leaky_relu,
                                dropout_rate_name='dropout_rate', has_shadow=has_shadow)
    network.add_layer_full_conn('FC4', 'H3', 'A', action_num, has_shadow=has_shadow)
    network.add_layer_full_conn('FC5', 'H3', 'V', 1, has_shadow=has_shadow)
    network.add_layer_duel_q('Duel_Q', 'V', 'A', q_name, has_shadow=has_shadow)



config = ActorDQN.Config(
    experience_size=experience_size,
    pick_len=train_seq_len,
    allow_short_seq=allow_short_seq,
    train_batch_size=train_batch_size,
    valid_sample_rate=valid_sample_rate,
    Q_network_func=Q_network_func,
    R_network_func=None,
    s_shape=[state_size],
    q_decay=q_decay,
    upd_shadow_period=upd_shadow_period,
    optimizer=Optimizer.Adam,
    double_dqn=True,
    # pick_selector_class=PickSelector.greedy_bin_heap
)


class enterprise_nnu:
    def __init__(self, num: int):
        self.enterprise = ActorDQN(config, num, clustered_devices)  # 生成num个mod
        self.epi_map = [None] * num
        self.epi_num = [0] * num
        self.epi_step = [0] * num
        self.epi_day = [0] * num  # epi_day记录企业存活天数
        self.last_state = [None] * num
        # self.enterprise.print_graphs_thread(logs_path, True)
            
    def run_enterprise(self, mod, state, day, is_train=False):  # enterprise_mod范围:[1, num]
        # if type(e_mod) == int:
        #     mod = e_mod
        #     mod = [mod]; state = [state]  # mod = [] state = [[...]]
        # else:
            # mod = e_mod.copy()
        # return [random.randint(0,9)]
        mod_num = len(mod)
        # =====准备工作=====#
        upd_mod = [];
        upd_state = [];
        upd_epi = [];
        upd_new_ep = []
        h_epi = [None] * mod_num  # 准备h_epi
        new_ep = [True] * mod_num  # 是否是新的ep
        state = list(state)
        for i in range(mod_num):
            state[i] = np.array(state[i])  # state准备就绪
            if mod[i] < 0:
                mod[i] = (-mod[i]) - 1
            else:
                mod[i] -= 0;
                new_ep[i] = False
            if self.last_state[mod[i]] is not None:
                upd_mod.append(mod[i])
                upd_epi.append(self.epi_map[mod[i]])
                upd_new_ep.append(new_ep[i])
                if new_ep[i]:
                    upd_state.append(self.last_state[mod[i]])
                else:
                    upd_state.append(state[i])
            self.last_state[mod[i]] = state[i]

        # record
        if upd_mod != []:
            self.env_upd(upd_mod, upd_state, upd_epi, upd_new_ep, day, is_train)

        # =====得到action=====#
        for i in range(mod_num):
            if not new_ep[i]:
                h_epi[i] = self.epi_map[mod[i]]
        h_epi, action = self.enterprise.episode_act(h_epi, state, mod)

        # =====更新self.epi_map self.epi_step,并对action进行处理=====#
        for i in range(mod_num):
            if new_ep[i]:
                self.epi_num[mod[i]] += 1
                self.epi_day[mod[i]] = 1
                self.epi_step[mod[i]] = 1
                self.epi_map[mod[i]] = h_epi[i]
            else:  # 旧企业
                self.epi_step[mod[i]] += 1
        return action

    def env_upd(self, mod, state, h_epi, new_ep, day, is_train):  # bank_mod范围:[1, num]
        mod_num = len(mod)

        # =====得到reward=====#
        reward = _custom_reward(state, new_ep,day)
        # =====把[state, action, reward, next_state]按序存入h_epi所属部分, 更新epi_map=====#
        for i in range(mod_num):
            # flag[i] += reward[i]

            h_epi[i] = self.enterprise.episode_feedback(h_epi[i], reward[i], state[i] if new_ep[i] else None)
            self.epi_map[mod[i]] = h_epi[i]
        # print(h_epi)
        # =====训练=====
        # is_train = False
        # for i in range(mod_num):
        #     if state[i][9] == 3:
        #         is_train = True
        #         break
        if is_train:
            train_ret = self.enterprise.model_train(learning_rate, dropout_rate_dict={'dropout_rate': dropout_rate})
            if isinstance(train_ret[0], tuple) and train_ret[0][0] == ActorDQN.FLAG_SHADOW_UPD:
                print('All enterprise_mod Shadow network updated.')
            # if (type(train_ret) == list) and (TRAIN_RET_SHADOW_UPD in train_ret):
            #     print('All enterprise_mod Shadow network updated.')

        # =====输出=====#
        for i in range(mod_num):
            if new_ep[i]:
                ep = self.epi_num[mod[i]]
                d = self.epi_day[mod[i]]
                step = self.epi_step[mod[i]]
                # print('enterprise_mod #', mod[i] + 1, ' ep #', ep, ' day #', d, ' lasts for ', step, ' steps')
                # print('enterprise_mod #', mod[i] + 1, ' ep #', ep, ' reward #', flag[i])
                # flag[i] = 0
                      
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


