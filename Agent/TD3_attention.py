import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import copy
import os
from Agent.Common.ExperienceReplay_NewLstm import Experience_Replay as ExpRep
from . import  Config
from Agent.RuningMeanStd import RunningMeanStd
import wandb
from stable_baselines3.common.utils import zip_strict
from typing import NamedTuple, Tuple
tranLock = True
isPercent = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()
        # d_model = s_dim  && s_dim % num_deads == 0
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Change shape to (seq_len, batch, d_model)
        attn_output, _ = self.multihead_attn(x, x, x)
        return attn_output.permute(1, 0, 2)  # Change back to (batch, seq_len, d_model)

class Actor(nn.Module):
    def __init__(self,s_dim,a_dim,a_bound):
        super(Actor, self).__init__()
        self.input_size =s_dim

        self.attention0 = SelfAttention(self.input_size,3)
        self.l1 = nn.Linear(self.input_size,128)
        # self.attention = SelfAttention(128, 4)
        self.l2 = nn.Linear(128,32)
        self.attention_ = SelfAttention(32, 2)
        self.l3 = nn.Linear(32,a_dim)
        self.a_bound = a_bound

    def forward(self,state):
        # (batch, seq_len, d_model)
        a = self.attention0(state)
        # a = F.tanh(self.l1(lstm_output))
        a = F.leaky_relu(self.l1(a))
        # a = F.leaky_relu(self.l1(state))
        # a = self.attention(a)
        a = F.leaky_relu(self.l2(a))
        a = self.attention_(a)
        a = F.tanh(self.l3(a))

        return self.a_bound * a

    def get_network_data(self):
        self.showl1 = Network(weight=self.l1.weight, bias=self.l1.bias)
        self.showl2 = Network(weight=self.l2.weight, bias=self.l2.bias)
        self.showl3 = Network(weight=self.l3.weight, bias=self.l3.bias)
        return self.showl1,self.showl2,self.showl3
        # return  self.showl2, self.showl3

class Critic(nn.Module):
    def __init__(self,s_dim,a_dim):
        super(Critic,self).__init__()
        self.input_size = s_dim+a_dim
        self.a_dim =a_dim
        self.s_dim = s_dim
        if self.input_size == 37:
            input_size = 40
        else:
            input_size = self.input_size

        self.attention01 = SelfAttention(input_size, 5)
        self.attention02 = SelfAttention(input_size, 5)
        self.attention1 = SelfAttention(128, 4)#5
        self.attention2 = SelfAttention(128, 4)#5
        # self.attention1_ = SelfAttention(32, 2)
        # self.attention2_ = SelfAttention(32, 2)
        # self.lstm_q1 = nn.LSTM(self.input,hidden_size=128)
        # self.lstm_q2 = nn.LSTM(self.input,hidden_size=128)
        #Q1
        self.l1 = nn.Linear(input_size,128)
        self.l2 = nn.Linear(128,32)
        self.l3 = nn.Linear(32,1)

        #Q2
        self.l4 = nn.Linear(input_size,128)
        self.l5 = nn.Linear(128,32)
        self.l6 = nn.Linear(32,1)

    def forward(self,state,action):
        sa = torch.cat((state,action), dim = -1)
        # 由于input size必须被heads整除，故此处需判断输入为企业还是银行
        # 若为企业 需要将输入补至40 [batch,seq,37] -> [batch,seq,40]
        # 银行输入为50则不需要补齐
        if self.input_size == 37:
            batch = sa.shape[0]
            seq = sa.shape[1]
            zeros = torch.zeros(batch, seq, 3)
            sa = torch.cat((sa,zeros), dim = -1)

        q1 = self.attention01(sa)
        q1 = F.leaky_relu(self.l1(q1))
        # q1 = F.leaky_relu(self.l1(sa))
        q1 = self.attention1(q1)
        q1 = F.leaky_relu(self.l2(q1))
        # q1 = self.attention1_(q1)
        q1 = self.l3(q1)

        q2 = self.attention01(sa)
        q2 = F.leaky_relu(self.l4(q2))
        # q2 = F.leaky_relu(self.l4(sa))
        q2 = self.attention2(q2)
        q2 = F.leaky_relu(self.l5(q2))
        # q2 = self.attention2_(q2)
        q2 = self.l6(q2)

        return q1, q2


    def Q1(self, state, action):

        sa = torch.cat((state,action), dim = -1)
        # 由于input size必须被heads整除，故此处需判断输入为企业还是银行
        # 若为企业 需要将输入补至40 [batch,seq,37] -> [batch,seq,40]
        # 银行输入为50则不需要补齐
        if self.input_size == 37:
            batch = sa.shape[0]
            seq = sa.shape[1]
            zeros = torch.zeros(batch, seq, 3)
            sa = torch.cat((sa, zeros), dim=-1)

        q1 = self.attention01(sa)
        q1 = F.leaky_relu(self.l1(q1))
        # q1 = F.leaky_relu(self.l1(sa))
        q1 = self.attention1(q1)
        q1 = F.leaky_relu(self.l2(q1))
        # q1 = self.attention1_(q1)
        q1 = self.l3(q1)

        return q1

    def get_network_data(self):
        self.showl1 = Network(weight=self.l1.weight, bias=self.l1.bias)
        self.showl2 = Network(weight=self.l2.weight, bias=self.l2.bias)
        self.showl3 = Network(weight=self.l3.weight, bias=self.l3.bias)
        self.showl4 = Network(weight=self.l4.weight, bias=self.l1.bias)
        self.showl5 = Network(weight=self.l5.weight, bias=self.l2.bias)
        self.showl6 = Network(weight=self.l6.weight, bias=self.l3.bias)
        return self.showl1,self.showl2,self.showl3,self.showl4,self.showl5,self.showl6
        # return self.showl2, self.showl3, self.showl4, self.showl6

class Network(NamedTuple):
    weight:torch.Tensor
    bias:torch.Tensor


class TD3(object):
    def __init__(self,config:Config):
        self.a_dim = config.action_dim
        self.s_dim = config.state_dim
        self.a_bound = config.action_bound
        self.scope = config.scope
        self.memory = Memory(capacity=config.MEMORY_CAPACITY, dims=2 * self.s_dim + self.a_dim + 1,pick_len=config.MAX_HIST_LEN)
        # self.memory = ReplayBuffer(max_size=config.MEMORY_CAPACITY,obs_dim = self.s_dim, act_dim = self.a_dim)
        self.LR_A = config.LEARNING_RATE_ACTOR
        self.LR_C = config.LEARNING_RATE_CRITIC
        self.LR_A_STABLE = config.LEARNING_RATE_ACTOR_STABLE
        self.LR_C_STABLE = config.LEARNING_RATE_CRITIC_STABLE
        self.LR_DECAY = config.LEARNING_RATE_DECAY
        self.LR_DECAY_TIME = config.LEARNING_RATE_DECAY_TIME
        self.GAMMA = config.REWARD_GAMMA
        self.TAU = config.SOFT_REPLACE_TAU
        self.BATCH_SIZE = config.BATCH_SIZE
        #self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.pointer = 0
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_dim))
        self.episode_temp = {}
        self.show_lar_a = 1
        # TD3参数
        self.is_delay = config.IS_ACTOR_UPDATE_DELAY
        self.is_double = config.IS_CRITIC_DOUBLE_NETWORK
        self.is_smooth = config.IS_QNET_SMOOTH_CRITIC
        self.update_cnt = 0 # 更新次数
        self.rms = RunningMeanStd(epsilon=0.0,shape=self.s_dim)
        self.is_rms=False
        self.tfRms = {}
        self.toshow = {}
        self.policy_noise =config.POLICY_NOISE
        #LSTM参数
        self.is_lstm = config.IS_LSTM
        self.hist_with_past_act = False
        self.MAX_HIST_LEN = config.MAX_HIST_LEN
        self.BATCH_SIZE_LSTM = config.BATCH_SIZE_LSTM
        self.update_every = config.UPDATE_EVERY
        self.discount = config.DISCOUNT
        self.last_lstm_states = None
        self.lstm_states = None
        self.hidden_states_a = None
        self.cell_states_a = None
        self.hidden_states_c = None
        self.cell_states_c = None
        self.episode_starts = None
        self.show_critic_loss = 0
        self.show_actor_loss = 0

        self.use_wandb = False
        if self.is_delay:
            self.policy_target_update_interval = config.ACTOR_UPDATE_DELAY_TIMES # 策略网络更新频率
        else:
            self.policy_target_update_interval = 1

        if self.is_smooth:
            self.eval_noise_scale = config.SMOOTH_NOISE  # 评估动作噪声缩放
        else:
            self.eval_noise_scale = 0.0  # 评估动作噪声缩放

        # Set seed
        self.set_global_seed(config.random_seed)

        self.var_init = config.VAR_INIT
        self.var_stable = config.VAR_STABLE
        self.var_drop_at = config.VAR_DROP_AT
        self.var_stable_at = config.VAR_STABLE_AT
        self.var_end_at = config.VAR_END_AT

        self.var = self.var_init
        self.lr_a = self.LR_A
        self.lr_c = self.LR_C


        #init Actor and Critic network(eval,target)
        self.actor= Actor(self.s_dim,self.a_dim,self.a_bound).to(device)
        self.actor_target= copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = self.lr_a)

        self.critic= Critic(self.s_dim,self.a_dim).to(device)
        self.critic_target= copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = self.lr_c)

        # hard_update 这里是师兄的trick
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(param.data)
            # Actor
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(param.data)



    def get_pointer(self):
        return self.pointer

    def choose_action_attn(self,h_epi,state):
        c = np.array(state)[np.newaxis,:]
        # rms 疑似不管用
        self.rms.update(c)
        if self.is_rms:
            state = (state - self.rms.mean)/(self.rms.var + 1e-5)
        # var 为探索目的而添加到动作空间的噪声方差
        self.var = self.var_init
        if self.pointer >self.var_stable_at:
            self.var = self.var_stable
        else:
            delta_step = self.pointer - self.var_drop_at
            if delta_step > 0 :
                self.var = self.var_init + delta_step * (self.var_stable - self.var_init)/(self.var_stable_at - self.var_drop_at)
        if h_epi is None:
            h_epi = self.memory.new_ep()
            self.episode_temp[h_epi] = dict()
        temp = self.episode_temp[h_epi]
        # state: [1,33] -> [1,1,33]
        state = torch.FloatTensor(state.reshape(1,1,-1)).to(device)
        with torch.no_grad():
            act =self.actor(state)

        action = act.cpu().detach().numpy().reshape(-1)
        # list = [act]
        # action=act[0][0]
        if isPercent:
            action = action + np.random.normal([0 for i in range(self.a_dim)], self.var)
            # action = np.clip(action,-self.a_bound,self.a_bound)
            for i in range(len(action)):
                if action[i] > self.a_bound:
                    action[i] = action[i] % self.a_bound
                if action[i] < -self.a_bound:
                    action[i] = action[i] % -self.a_bound
        else:
            action = np.clip(action + np.random.normal([0 for i in range(self.a_dim)], self.var), 0.001,
                             100000000)  # 固定值
        # showl1,showl2,showl3 = self.actor.get_network_data()

        return h_epi, action

    def choose_action(self,h_epi,state):
        c = np.array(state)[np.newaxis,:]
        # rms 疑似不管用
        self.rms.update(c)
        if self.is_rms:
            state = (state - self.rms.mean)/(self.rms.var + 1e-5)

        self.var = self.var_init
        if self.pointer >self.var_stable_at:
            self.var = self.var_stable
        else:
            delta_step = self.pointer - self.var_drop_at
            if delta_step > 0 :
                self.var = self.var_init + delta_step * (self.var_stable - self.var_init)/(self.var_stable_at - self.var_drop_at)
        if h_epi is None:
            h_epi = self.memory.new_ep()
            self.episode_temp[h_epi] = dict()
        temp = self.episode_temp[h_epi]

        state=state.reshape(1,-1)
        list = [self.actor(state)]
        action = list[0][0]
        if isPercent:
            action = action.cpu().detach() + np.random.normal([0 for i in range(self.a_dim)], self.var)
            for i in range(len(action)):
                if action[i] > self.a_bound:
                    action[i] = action[i] % self.a_bound
                if action[i] < -self.a_bound:
                    action[i] = action[i] % -self.a_bound
        else:
            action = np.clip(action.cpu().detach() + np.random.normal([0 for i in range(self.a_dim)], self.var), 0.001,
                             100000000)  # 固定值
        return h_epi,action.numpy()

    def episode_feedback(self, h_epi, state, action, reward, final_state):
        if final_state is not None:
            final_state = np.zeros(len(final_state))
            # final_state = final_state/1000
        self.pointer += 1
        if self.pointer%100 == 0:
            print(self.mark())

        # self.memory.store_transition(obs = state, act=action, rew=reward, next_obs=final_state, done=done)
        ret_h_epi = self.memory.store_transition(h_epi, state, action, reward, final_state)

        return ret_h_epi

    def mark(self):
        return "_______________________________________@(@*#(@#*( " + str(self.pointer) + " " + str(
            self.var) + " " + str(self.show_lar_a) + "_______________________________24$@A#@$"

    def learn(self):

        if self.pointer < 3000:
            return 0
        self.update_cnt += 1
        if self.pointer > 8000 and self.pointer%100 == 0:
            a = 1

        # 从replay buffer（通过memory调用）中随机采样的样本数据
        batch = self.memory.sample(self.BATCH_SIZE_LSTM, s_dim = self.s_dim)
        # batch = self.memory.sample(self.BATCH_SIZE, s_dim=self.s_dim)
        batch = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
        batch = {k: v.to(device) for k, v in batch.items()}
        b_s = batch['s']
        b_a = batch['a']
        b_s_ = batch['s_']
        b_r = batch['r']

        # 根据当前训练步数（pointer）计算Critic和Actor网络的学习率lr_c，lr_a
        self.lr_a = max(self.LR_A_STABLE, self.LR_A * np.power(self.LR_DECAY, ((self.pointer - 3000) / self.LR_DECAY_TIME)))
        self.lr_c = max(self.LR_C_STABLE, self.LR_C * np.power(self.LR_DECAY, ((self.pointer - 3000) / self.LR_DECAY_TIME)))
        self.show_lar_a = self.lr_a
        # update
        if (not tranLock) or self.pointer < self.var_end_at:

            with torch.no_grad():
                # 计算扰动噪声后的动作a_
                if self.is_smooth:
                    action_= self.actor_target(b_s_)

                    sample = torch.distributions.Normal(0., 1.)
                    a_dim = (self.a_dim,)
                    noise = torch.clamp(sample.sample(a_dim) * self.eval_noise_scale, -2 * self.eval_noise_scale,
                                        2 * self.eval_noise_scale)
                    action_with_noise = (action_ + noise).clamp(-self.a_bound, self.a_bound)

                    # noise = (torch.randn_like(action_) * self.policy_noise).clamp(-self.a_bound,self.a_bound)
                    # action_with_noise = ( action_+ noise).clamp(-self.a_bound,self.a_bound)

                # 计算target Q 值
                target_Q1,target_Q2= self.critic_target(b_s_, action_with_noise)
                # target_Q (batch_size*pick_len,1) --> (batch_size,pick_len)
                # target_Q1 = target_Q1.reshape(self.BATCH_SIZE_LSTM, -1)
                # target_Q2 = target_Q2.reshape(self.BATCH_SIZE_LSTM,-1)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = target_Q.reshape(self.BATCH_SIZE_LSTM,-1)

                # change 0513 only use last step of Q value and reward
                # last_b_r = b_r[:,-1].unsqueeze(1)
                # last_TQ1 = target_Q1[:,-1].unsqueeze(1)
                # last_TQ2 = target_Q2[:,-1].unsqueeze(1)
                # last_target_Q = torch.min(last_TQ1, last_TQ2)
                # last_target_Q = last_target_Q * self.GAMMA + last_b_r
                target_Q = b_r + self.GAMMA * target_Q

            # 获得当前batch Q estimates
            current_Q1, current_Q2 = self.critic(b_s,b_a)

            # showl1, showl2, showl3 ,showl4, showl5, showl6 = self.critic.get_network_data()

            # current_Q1 (4000,1) -->(batch_size,pick_len)
            current_Q1 = current_Q1.reshape(self.BATCH_SIZE_LSTM,-1)
            current_Q2 = current_Q2.reshape(self.BATCH_SIZE_LSTM,-1)

            # change 0513 only use last step of Q value and reward
            # last_current_Q1 = current_Q1[:,-1].unsqueeze(1)
            # last_current_Q2 = current_Q2[:,-1].unsqueeze(1)
            critic_loss = F.mse_loss(current_Q1,target_Q) + F.mse_loss(current_Q2,target_Q)

            # 计算critic loss = td - error
            # critic_loss = F.mse_loss(current_Q1,target_Q) + F.mse_loss(current_Q2,target_Q)
            self.show_critic_loss = critic_loss

            if self.use_wandb:
                wandb.log({'loss/critic':critic_loss})
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 延迟策略更新
            if self.update_cnt % self.policy_target_update_interval == 0:
            # if j % self.policy_target_update_interval == 0:

                action= self.actor(b_s)
                Q1_critic=self.critic.Q1(b_s,action)
                Q1_critic = Q1_critic.reshape(self.BATCH_SIZE_LSTM,-1)
                actor_loss = -Q1_critic.mean()

                self.show_actor_loss = actor_loss

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()


                # soft update
                #   Critic
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
                #   Actor
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        # 只计算critic 的loss 不进行网络更新
        else:
            with torch.no_grad():
                # 计算扰动噪声后的动作a_
                if self.is_smooth:
                    action_ = self.actor_target(b_s_)

                    sample = torch.distributions.Normal(0., 1.)
                    a_dim = (self.a_dim,)
                    # sample_ = sample.sample(a_dim)
                    noise = torch.clamp(sample.sample(a_dim) * self.eval_noise_scale, -2 * self.eval_noise_scale,
                                        2 * self.eval_noise_scale)
                    action_with_noise = (action_ + noise).clamp(-self.a_bound, self.a_bound)

                    # noise = (torch.randn_like(action_) * self.policy_noise).clamp(-self.a_bound,self.a_bound)
                    # action_with_noise = ( action_+ noise).clamp(-self.a_bound,self.a_bound)

                # 计算target Q 值
                target_Q1, target_Q2 = self.critic_target(b_s_, action_with_noise)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = target_Q.reshape(self.BATCH_SIZE_LSTM,-1)
                target_Q = b_r + self.GAMMA * target_Q

            # 获得当前batch Q estimates
            current_Q1, current_Q2 = self.critic(b_s,b_a)
            current_Q1 = current_Q1.reshape(self.BATCH_SIZE_LSTM,-1)
            current_Q2 = current_Q2.reshape(self.BATCH_SIZE_LSTM,-1)

            # 计算critic loss = td - error
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.show_critic_loss = critic_loss

        # return critic_loss


        # 将经验存储到缓冲池（memory）中
        # 参数：时间步数（或者称作episode的索引）、当前状态、选择的动作、获得的奖励以及下一个状态
    def store_transition(self, s, a, r, s_,done):
        # transition = np.hstack((s, a, [r], s_))
        # index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # self.memory[index, :] = transition
        self.pointer += 1  # 存储了一条新经验
        self.memory.store_transition(s, a, r, s_,done)  # 将当前状态、动作、奖励和下一个状态作为一条完整的经验（transition）存储到经验缓冲池中

    def translate(self,data):
        for d in range(len(data)):
            data[d] = data[d]/1000
        return np.array(data)

    def set_global_seed(self,seed):
        # pytorch_seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # python_seed
        random.seed(seed)
        #NumPy_seed
        np.random.seed(seed)

        random.seed(seed)

    def get_var(self):
        return self.var

    def get_loss(self):
        return self.show_critic_loss, self.show_actor_loss




class Memory(object):
    def __init__(self, capacity, dims,pick_len):
        self.capacity = capacity
        self.pick_len = pick_len
        self.data = np.zeros((capacity, dims))
        self.pointer = 0
        self.exp_rep = ExpRep(capacity, pick_len = self.pick_len, allow_short_seq = True)
        self.sample_rate=0.03
        self.size = 0


    def store_transition(self,h_epi, s, a, r, s_):
        # 在此处将array转为c_int64
        a = self.exp_rep.encoded_actions(action=a)
        self.size = min(self.size+1,self.capacity)
        return self.exp_rep.record(h_epi,s,a,r,s_)


    def sample(self, n, s_dim):
        # assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        # indices = np.random.choice(self.capacity, size=n)
        # return self.data[indices, :]
        return self.exp_rep.get_random_batch(n,self.sample_rate)

    def new_ep(self):
        return self.exp_rep.new_episode()

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.02, theta=1, dt=1e-5, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self,sigma):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

def is_list(d):
    return isinstance(d, list)