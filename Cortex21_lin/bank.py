'''
ep = 企业, mod = 企业家, 一个episode = 一个企业从生到死
bank = bank_nnu(num:int)//传入企业家的个数,生成num个mod
bank.run_bank(bank_mod, state)//传入经营的企业家编号&企业状态,输出[银行的贷款额度, 银行的利息]
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

warnings.filterwarnings('ignore')

io_path = 'io/'
ex_path = io_path + 'bank_nnu/'
logs_path = ex_path + 'logs'
session_path = ex_path + 'session'
model_filename = ex_path + 'model'

clustered_devices = None


def _custom_reward(state, new_ep=None):
    reward = []
    for i in range(len(state)):
        r = state[i][1]
        # r = state[i][1] + 100*state[i][3]
        if new_ep[i]:
            r = -r
        reward.append(r)
    return reward


experience_size = 50000   # 经验库的大小
train_seq_len = 1  # 样本的时间序列长度
allow_short_seq = False  # 是否允许取序列长度不足seq len所指定的长度
train_batch_size = 200  # 一次取多少样本出来训练 LSTM 10 else 200
valid_sample_rate = 0.03  # 当经验库中样本达到 batch size * rate时，反复抽取经验库中的经验进行训练
feeding_buffer = 0  # 用于异步进行计算q target，如果这个值不是0，
                    # 则会建立专门线程，将抽取的batch先计算q-target，
                    # train的时候直接使用，这个值你可以理解为预先计算多少个batch的q-target

state_size = 8  # 神经网络的输入数据state有8个dimension
action_num = 10  # action的取值范围[0,action_num - 1]

q_decay = 0.95
upd_shadow_period = 100  # 训练多少次后更新 q_target

learning_rate = 0.01
dropout_rate = 0.02

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

#lstm版本
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


config = ActorDQN.Config(              # 配置
    experience_size=experience_size,
    pick_len=train_seq_len,
    allow_short_seq=allow_short_seq,
    train_batch_size=train_batch_size,
    valid_sample_rate=valid_sample_rate,
    Q_network_func=Q_network_func,
    R_network_func=None,
    s_shape=[state_size],  # s_shape=[8]
    q_decay=q_decay,
    upd_shadow_period=upd_shadow_period,
    optimizer=Optimizer.Adam,
    double_dqn=True,
    # pick_selector_class=PickSelectorClass.greedy_bin_heap
)


class bank_nnu:
    def __init__(self, num: int):
        self.bank = ActorDQN(config, num, clustered_devices)  # 生成num个mod
        self.epi_map = [None] * num  # 把mod和ep形成映射 epi_map = [mod0_ep, mod1_ep, mod2_ep] mod当前运营的ep在h_epi中的编号
        self.epi_num = [0] * num  # 银行家i运营银行数
        self.epi_step = [0] * num  # mod的ep的决策步数，epi_step = [mod0_ep_step,mod1_ep_step,...,mod[num]_ep_step]
        self.action = [None] * 2  # 决策贷款额度和利息两个动作
        self.last_state = [None] * num  # 银行家i的银行的上一个state
        # self.bank.print_graphs_thread(logs_path, True)  # 生成神经网络的初始结构图，保存在io/bank_nnu/logs，用tensorboard查看

    def run_bank(self, b_mod, state):  # bank_mod范围:[1, num]
        if type(b_mod) == int:
            mod = b_mod
            mod = [mod]; state = [state]
        else:
            mod = b_mod.copy()
        mod_num = len(mod)  # 银行家个数
        state = list(state)

        # =====准备工作=====#
        upd_mod = []; upd_state = []; upd_epi = []; upd_new_ep = []
        new_ep = [True] * mod_num
        h_epi = [None] * mod_num  # 句柄，新公司有新编号（0,1,2,3.....)
        for i in range(mod_num):
            state[i] = list(state[i])
            state[i].insert(0, 1)  # 在state[i]的第0个位置插入1,1表示决策贷款额度
            state[i] = np.array(state[i])  # state准备就绪
            if mod[i] < 0:  # 银行i是新的
                mod[i] = (-mod[i]) - 1  # mod[1,2,3] -> mod[0,1,2]
            else:  # 银行i是旧的
                mod[i] = mod[i] - 1; new_ep[i] = False  # mod[i]的正旧负新 -> 用new_ep[i]表示
            if self.last_state[mod[i]] is not None:  # 若银行i的上个state是存在的  只有day0时出现last_state[mod[i]] is None
                if new_ep[i]:
                    upd_state.append(self.last_state[mod[i]])  # 新银行则用上一个银行的最后状态作为更新
                else:
                    upd_state.append(state[i])  # 旧银行用传入的state作为更新
                upd_mod.append(mod[i])
                upd_epi.append(self.epi_map[mod[i]])  # upd_epi = self.epi_map = h_epi
                upd_new_ep.append(new_ep[i])

        # record
        if upd_mod != []:  # 若有银行存在
            self.env_upd(upd_mod, upd_state, upd_epi, upd_new_ep)

        # ========第一次训练，并保存action========#
        for i in range(mod_num):
            if not new_ep[i]:  # 旧的银行
                h_epi[i] = self.epi_map[mod[i]]
        # =====得到action,保存到self.action[0]中=====#
        h_epi, self.action[0] = self.bank.episode_act(h_epi, state, mod)
        # print(h_epi,self.epi_map,upd_state)
        # 第一次输入h_epi得到了更新，第二次数据输入就不用考虑h_epi,可以直接用
        # =====把第一次训练的[s,a,r,s']存入experience中=====#
        # =====得到reward=====#
        reward = _custom_reward(state, [False]*mod_num)
        # =====把[state, action, reward, next_state]按序存入h_epi所属部分=====#
        h_epi = self.bank.episode_feedback(h_epi, reward)
        for i in range(mod_num):
            self.epi_map[mod[i]] = h_epi[i]
        # print(h_epi,self.epi_map)
        # ========第二次训练，并保存action========#
        # =====准备state=====#
        for i in range(mod_num):
            state[i][0] = 2  # 2表示决策利息
            self.last_state[mod[i]] = state[i]
        # =====得到action,保存到self.action[1]中=====#
        h_epi, self.action[1] = self.bank.episode_act(h_epi, state, mod)  # 把state输入网络，决策利息
        # print(h_epi,self.epi_map)
        # =====更新self.epi_map self.epi_step=====#
        for i in range(mod_num):
            if new_ep[i]:
                self.epi_step[mod[i]] = 1
                self.epi_num[mod[i]] += 1
                self.epi_map[mod[i]] = h_epi[i]  # self.epi_map保存该loop最后的h_epi 下个loop用来赋值给upd_epi
            else:
                self.epi_step[mod[i]] += 1

        return np.array(self.action, dtype=float)

    def env_upd(self, mod, state, h_epi, new_ep):  # bank_mod范围:[1, num]  这里的h_epi是upd_epi
        mod_num = len(mod)

        # =====得到reward=====#
        reward = _custom_reward(state, new_ep)
        # =====把[state, action, reward, next_state]按序存入h_epi所属部分,更新epi_map=====#
        for i in range(mod_num):
            h_epi[i] = self.bank.episode_feedback(h_epi[i], reward[i], state[i] if new_ep[i] else None)  # record
            self.epi_map[mod[i]] = h_epi[i]

        # =====训练=====#
        train_ret = self.bank.model_train(learning_rate, dropout_rate_dict={'dropout_rate': dropout_rate})
        # if isinstance(train_ret, tuple) and train_ret[0] == ActorDQN.FLAG_SHADOW_UPD:
        #     print('Shadow network updated.')
        if isinstance(train_ret[0], tuple) and train_ret[0][0] == ActorDQN.FLAG_SHADOW_UPD:
            print('All bank_mod Shadow network updated.')

        # =====输出=====#
        for i in range(mod_num):
            if new_ep[i]:
                ep = self.epi_num[mod[i]]
                step = self.epi_step[mod[i]]
                print('bank_mod #', mod[i] + 1, ' ep #', ep, ' lasts for ', step, ' steps')
                  
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


