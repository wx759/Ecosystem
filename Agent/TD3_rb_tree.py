import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import copy
import os
from Agent.Common.ExperienceReplay_TD3 import Experience_Replay as ExpRep
from . import  Config
from Agent.RuningMeanStd import RunningMeanStd
# from Agent.RuningMeanStd import TfRunningMeanStd
import wandb
tranLock = True
isPercent = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Actor(nn.Module):
    def __init__(self,s_dim,a_dim,a_bound):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(s_dim,128)
        self.l2 = nn.Linear(128,32)
        self.l3 = nn.Linear(32,a_dim)

        self.a_bound = a_bound

    def forward(self,state):
        state = torch.FloatTensor(state).to(device)
        a = F.tanh(self.l1(state))
        a = F.leaky_relu(self.l2(a))
        return self.a_bound * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self,s_dim,a_dim):
        super(Critic,self).__init__()

        #Q1
        self.l1 = nn.Linear(s_dim+a_dim,128)
        self.l2 = nn.Linear(128,32)
        self.l3 = nn.Linear(32,1)

        #Q2
        self.l4 = nn.Linear(s_dim+a_dim,128)
        self.l5 = nn.Linear(128,32)
        self.l6 = nn.Linear(32,1)

    def forward(self,state,action):
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)

        sa = torch.cat([state,action],1)

        q1 = F.leaky_relu(self.l1(sa))
        q1 = F.leaky_relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.leaky_relu(self.l4(sa))
        q2 = F.leaky_relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1,q2
    def Q1(self,state,action):
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)

        sa = torch.cat([state,action],1)

        q1 = F.leaky_relu(self.l1(sa))
        q1 = F.leaky_relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class TD3(object):
    def __init__(self,config:Config):
        self.a_dim = config.action_dim
        self.s_dim = config.state_dim
        self.a_bound = config.action_bound
        self.scope = config.scope
        self.memory = Memory(capacity=config.MEMORY_CAPACITY, dims=2 * self.s_dim + self.a_dim + 1)
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
        # self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_dim))
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
        self.pre_lstm_hid_sizes = (128,)
        self.lstm_hid_sizes = (128,)
        self.after_lstm_hid_size = (128,)
        self.cur_feature_hid_sizes = (128,)
        self.post_comb_hid_sizes = (128,)
        self.hist_with_past_act = False
        self.discount = config.DISCOUNT
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
        self.actor = Actor(self.s_dim,self.a_dim,self.a_bound).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = self.lr_a)

        self.critic = Critic(self.s_dim,self.a_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = self.lr_c)
        # hard_update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(param.data)
            # Actor
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(param.data)



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
                self.var = self.var_init +delta_step * (self.var_stable - self.var_init)/(self.var_stable_at - self.var_drop_at)
        if h_epi is None:
            h_epi = self.memory.new_ep()
            self.episode_temp[h_epi] = dict()
        temp = self.episode_temp[h_epi]
        #
        '''
        to_run = [self.actor(state[np.newaxis,:])]
        list = to_run
        action = list[0][0]

        if isPercent:
            action = action.cpu().detach() + np.random.normal([0 for i in range(self.a_dim)],self.var)
            for i in range(len(action)):
                if action[i] > self.a_bound:
                    action[i] = action[i] % self.a_bound
                if action[i] < -self.a_bound:
                    action[i] = action[i] % -self.a_bound
        else:
            action = np.clip(action + np.random.normal([0 for i in range(self.a_dim)],self.var), 0.001, 100000000) # 固定值

        '''
        list = [self.actor(state.reshape(1,-1))]
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

    def episode_feedback(self,h_epi, state, action, reward, final_state):
        if final_state is not None:
            final_state = np.zeros(len(final_state))
        self.pointer += 1
        if self.pointer%100 == 0:
            print(self.mark())

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
        b_M = self.memory.sample(self.BATCH_SIZE)
        # b_M = self.memory.sample(2)
        # 数据处理（归一化） ，b_s_rm,b_s__rm 当前状态和下一个状态的处理后的观测数据
        if self.is_rms:
            b_s_rm = (b_M[0] - self.rms.mean) / (self.rms.var + 1e-5)
            b_s__rm = (b_M[3] - self.rms.mean) / (self.rms.var + 1e-5)
        else:
            b_s_rm = b_M[0]
            b_s__rm = b_M[3]
        b_s = b_s_rm.reshape(-1, self.s_dim)

        # 根据网络设定获取当前状态b_s，动作b_a，奖励b_r，下一个状态b_s_的数据，进行数据处理
        # b_a = b_M[1].reshape(-1,action_dim)/10000-2.
        b_a = np.array(b_M[1]).reshape(-1, self.a_dim)
        b_r = b_M[2].reshape(-1, 1)
        b_s_ = b_s__rm.reshape(-1, self.s_dim)
        # 根据当前训练步数（pointer）计算Critic和Actor网络的学习率lr_c，lr_a
        # 其实没什么用
        # self.lr_a = max(self.LR_A_STABLE, self.LR_A * np.power(self.LR_DECAY, ((self.pointer - 3000) / self.LR_DECAY_TIME)))
        # self.lr_c = max(self.LR_C_STABLE, self.LR_C * np.power(self.LR_DECAY, ((self.pointer - 3000) / self.LR_DECAY_TIME)))
        # self.show_lar_a = self.lr_a
        if (not tranLock) or self.pointer < self.var_end_at:

            with torch.no_grad():
                # 计算扰动噪声后的动作a_ (没有用师兄原本的噪声， 用的td3 的噪声
                if self.is_smooth:

                    # noise = (torch.randn_like(torch.FloatTensor(b_a)) * self.policy_noise).clamp(-self.a_bound,self.a_bound)
                    # action_with_noise = (self.actor_target(b_s_) + noise).clamp(-self.a_bound,self.a_bound)

                    sample = torch.distributions.Normal(0., 1.)
                    a_dim = (self.a_dim,)
                    sample_ = sample.sample(a_dim)
                    noise = torch.clamp(sample_ * self.eval_noise_scale,-2 * self.eval_noise_scale,2 * self.eval_noise_scale)
                    action_with_noise = (self.actor_target(b_s_)+noise).clamp(-self.a_bound,self.a_bound)
                # 计算target Q 值
                target_Q1,target_Q2 = self.critic_target(b_s_,action_with_noise)
                target_Q = torch.min(target_Q1, target_Q2)
                # target_Q = torch.tensor(b_r) + self.GAMMA * target_Q * self.discount
                target_Q = torch.tensor(b_r) + self.GAMMA * target_Q

            # 获得当前batch Q estimates
            current_Q1,current_Q2 = self.critic(b_s,b_a)
            # 计算critic loss = td - error
            critic_loss = F.mse_loss(current_Q1,target_Q) +F.mse_loss(current_Q2,target_Q)
            wandb.log({"loss/critic":critic_loss})

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


            # 延迟策略更新
            if self.update_cnt % self.policy_target_update_interval == 0:
                actor_loss = -self.critic.Q1(b_s,self.actor(b_s)).mean()

                wandb.log({"loss/actor": actor_loss})
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # soft update
                #   Critic
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param)
                #   Actor
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param)
        # 只计算critic 的loss 不进行网络更新
        else:
            '''with torch.no_grad():
                # 计算扰动噪声后的动作a_
                if self.is_smooth:
                    sample = torch.distributions.Normal(0., 1.)
                    # 数据格式处理
                    a_dim = [self.a_dim]
                    a_dim = torch.tensor(a_dim)
                    torch.unsqueeze(a_dim, 0)
                    x = sample.sample(a_dim)
                    noise = torch.clamp(x * self.eval_noise_scale,
                                        -2 * self.eval_noise_scale,
                                        2 * self.eval_noise_scale)  # torch.clamp(x,min,max)
                    noise_a_ = torch.clamp(self.actor_target(b_s_) + noise, self.a_bound, self.a_bound)
                else:
                    noise_a_ = self.actor_target(b_s_)
                # 计算target Q 值
                target_Q1 = self.critic_target.Q1(b_s_, noise_a_)
                target_Q2 = self.critic_target.Q2(b_s_, noise_a_)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = torch.tensor(b_r) + self.GAMMA * target_Q
                # 获得当前batch Q estimates'
            '''
            with torch.no_grad():
                # 计算扰动噪声后的动作a_ (没有用师兄原本的噪声， 用的td3 的噪声
                if self.is_smooth:
                    # noise = (torch.randn_like(torch.FloatTensor(b_a)) * self.policy_noise).clamp(-self.a_bound,self.a_bound)
                    # action_with_noise = (self.actor_target(b_s_) + noise).clamp(-self.a_bound,self.a_bound)

                    sample = torch.distributions.Normal(0., 1.)
                    a_dim = (self.a_dim,)
                    sample_ = sample.sample(a_dim)
                    noise = torch.clamp(sample_ * self.eval_noise_scale, -2 * self.eval_noise_scale,
                                        2 * self.eval_noise_scale)
                    action_with_noise = (self.actor_target(b_s_) + noise).clamp(-self.a_bound, self.a_bound)

                # 计算target Q 值
                target_Q1,target_Q2 = self.critic_target(b_s_,action_with_noise)
                target_Q = torch.min(target_Q1, target_Q2)
                # target_Q = torch.tensor(b_r) + self.GAMMA * target_Q * self.discount
                target_Q = torch.tensor(b_r) + self.GAMMA * target_Q

            # 获得当前batch Q estimates
            current_Q1,current_Q2 = self.critic(b_s,b_a)

            # 计算critic loss = td - error
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            wandb.log({"loss/critic": critic_loss})
        return critic_loss

        # 将经验存储到缓冲池（memory）中
        # 参数：时间步数（或者称作episode的索引）、当前状态、选择的动作、获得的奖励以及下一个状态
    def store_transition_rbtree(self, h_epi, s, a, r, s_):

        self.memory.store_transition_rbtree(h_epi, s, a, r, s_)  # 将当前状态、动作、奖励和下一个状态作为一条完整的经验（transition）存储到经验缓冲池中

    def store_transition(self, h_epi, s, a, r, s_):
        # transition = np.hstack((s, a, [r], s_))
        # index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # self.memory[index, :] = transition
        self.pointer += 1  # 存储了一条新经验
        self.memory.store_transition(h_epi, s, a, r, s_)  # 将当前状态、动作、奖励和下一个状态作为一条完整的经验（transition）存储到经验缓冲池中

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

        # 设置python的hash随机种子
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        # 设置PyTorch使用的算法为确定性算法
        torch.backends.cudnn.benchmark = False



class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0
        self.exp_rep = ExpRep(capacity, 1, False,True,True)
        self.sample_rate=0.03
        self.PickSelector = 1

    def store_transition(self,h_epi, s, a, r, s_):

        return self.exp_rep.record(h_epi,s,a,r,s_)

    def store_transition_rbtree(self,h_epi, s, a, r ,s_):
        epi = h_epi - 1 + 80000
        return self.exp_rep.record(epi,s,a,r,s_)


    def sample(self, n):
        # assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        # indices = np.random.choice(self.capacity, size=n)
        # return self.data[indices, :]
        h_ps = self.exp_rep.new_pick_selector(pick_selector_class = 'greedy_rb_tree')
        return self.exp_rep.get_random_batch(n,self.sample_rate,h_ps=h_ps)

    def new_ep(self):
        return self.exp_rep.new_episode()
