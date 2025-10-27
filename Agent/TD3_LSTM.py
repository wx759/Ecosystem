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
import wandb
tranLock = True
isPercent = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Actor(nn.Module):
    def __init__(self,s_dim,a_dim,a_bound):
        super(Actor, self).__init__()
        self.input_size = s_dim + a_dim
        self.lstm = nn.LSTM(self.input_size,128,batch_first=True) # input_size = s+a hidden_size = 128 单向单层lstm
        self.l1 = nn.Linear(s_dim,128)
        self.l2 = nn.Linear(256,32)
        self.l3 = nn.Linear(32,a_dim)

        self.a_bound = a_bound

    def forward(self,state,h_state,h_action,h_seg_len):

        # 数据格式处理
        if (h_state is None) or (h_action is None) or (h_seg_len is None):
            h_state = torch.zeros(1, 1, self.s_dim).to(device)
            h_action = torch.zeros(1, 1, self.a_dim).to(device)
            h_seg_len = torch.zeros(1).to(device)


        # LSTM
        tmp_h_seg_len = copy.deepcopy(h_seg_len)
        tmp_h_seg_len[h_seg_len == 0] = 1

        x = torch.cat([h_state,h_action],dim = -1)
        x , (lstm_hidden_state, lstm_cell_state) = self.lstm(x)

        hist_out = torch.gather(x, 1,(tmp_h_seg_len - 1).view(-1, 1).repeat(1,128).unsqueeze(1).long()).squeeze(1)

        # current Feature
        a = F.tanh(self.l1(state))
        a = torch.cat([hist_out, a], dim=-1)
        a = F.leaky_relu(self.l2(a))
        return self.a_bound * torch.tanh(self.l3(a)),hist_out



class Critic(nn.Module):
    def __init__(self,s_dim,a_dim):
        super(Critic,self).__init__()
        self.lstm_q1 = nn.LSTM(s_dim,128,batch_first=True)
        self.lstm_q2 = nn.LSTM(s_dim,128,batch_first=True)
        #Q1
        self.l1 = nn.Linear(s_dim+a_dim,128)
        self.l2 = nn.Linear(256,32)
        self.l3 = nn.Linear(32,1)

        #Q2
        self.l4 = nn.Linear(s_dim+a_dim,128)
        self.l5 = nn.Linear(256,32)
        self.l6 = nn.Linear(32,1)

    def forward(self,state,action,h_state,h_seg_len):
        x = h_state

        sa = torch.cat([state,action],1)

        # LSTM
        tmp_h_seg_len = copy.deepcopy(h_seg_len)
        tmp_h_seg_len[h_seg_len == 0] = 1
        x1, (lstm_hidden_state, lstm_cell_state) = self.lstm_q1(x)
        hist_out1 = torch.gather(x1, 1, (tmp_h_seg_len - 1).view(-1, 1).repeat(1, 128).unsqueeze(1).long()).squeeze(1)
        x2,(lstm_hidden_state,lstm_cell_state) =self.lstm_q2(x)
        hist_out2 = torch.gather(x2, 1, (tmp_h_seg_len - 1).view(-1, 1).repeat(1, 128).unsqueeze(1).long()).squeeze(1)

        # current Feature
        q1 = F.leaky_relu(self.l1(sa))
        q1 = torch.cat([hist_out1, q1], dim=-1)
        q1 = F.leaky_relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.leaky_relu(self.l4(sa))
        q2 = torch.cat([hist_out2,q2],dim=-1)
        q2 = F.leaky_relu(self.l5(q2))
        q2 = self.l6(q2)

        return torch.squeeze(q1, -1), hist_out1,torch.squeeze(q2, -1), hist_out2

    def Q1(self,state,action,h_state,h_seg_len):

        x = h_state

        sa = torch.cat([state, action], 1)

        # LSTM
        tmp_h_seg_len = copy.deepcopy(h_seg_len)
        tmp_h_seg_len[h_seg_len == 0] = 1
        x1, (lstm_hidden_state, lstm_cell_state) = self.lstm_q1(x)
        hist_out1 = torch.gather(x1, 1, (tmp_h_seg_len - 1).view(-1, 1).repeat(1, 128).unsqueeze(1).long()).squeeze(1)

        # current Feature
        q1 = F.leaky_relu(self.l1(sa))
        q1 = torch.cat([hist_out1, q1], dim=-1)
        q1 = F.leaky_relu(self.l2(q1))
        q1 = self.l3(q1)
        return torch.squeeze(q1, -1), hist_out1



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

        # self.use_wandb = True
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
        self.show_critic_loss = 0
        self.show_actor_loss = 0


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

    def choose_action_lstm(self,h_epi,state,state_buff,action_buff,state_buff_len):
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
        #

        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        h_s = torch.tensor(state_buff).view(1, state_buff.shape[0], state_buff.shape[1]).float().to(device)
        h_a = torch.tensor(action_buff).view(1, action_buff.shape[0], action_buff.shape[1]).float().to(device)
        h_l = torch.tensor([state_buff_len]).float().to(device)
        with torch.no_grad():
            act , _ =self.actor(state,h_s,h_a,h_l)
        act = act.cpu().detach().numpy()
        list = [act]
        action=list[0][0]
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

    def episode_feedback(self, h_epi, state, action, reward, final_state):
        if final_state is not None:
            final_state = np.zeros(len(final_state))
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
        batch = self.memory.sample(self.BATCH_SIZE_LSTM,a_dim = self.a_dim)
        batch = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
        batch = {k: v.to(device) for k, v in batch.items()}
        b_s = batch['s']
        b_a = batch['a']
        b_s_ = batch['s_']
        b_r = batch['r']
        #lstm
        h_s = batch['h_s']
        h_a = batch['h_a']
        h_s_ = batch['h_s_']
        h_a_ = batch['h_a_']
        h_s_len = batch['h_s_len']



        # 根据当前训练步数（pointer）计算Critic和Actor网络的学习率lr_c，lr_a
        self.lr_a = max(self.LR_A_STABLE, self.LR_A * np.power(self.LR_DECAY, ((self.pointer - 3000) / self.LR_DECAY_TIME)))
        self.lr_c = max(self.LR_C_STABLE, self.LR_C * np.power(self.LR_DECAY, ((self.pointer - 3000) / self.LR_DECAY_TIME)))
        self.show_lar_a = self.lr_a
        # update
        if (not tranLock) or self.pointer < self.var_end_at:

            with torch.no_grad():
                # 计算扰动噪声后的动作a_ (没有用师兄原本的噪声， 用的td3 的噪声
                if self.is_smooth:
                    action_, _ = self.actor_target(b_s_, h_s_, h_a_, h_s_len)

                    sample = torch.distributions.Normal(0., 1.)
                    a_dim = (self.a_dim,)
                    # sample_ = sample.sample(a_dim)
                    noise = torch.clamp(sample.sample(a_dim) * self.eval_noise_scale, -2 * self.eval_noise_scale,
                                        2 * self.eval_noise_scale)
                    action_with_noise = (action_ + noise).clamp(-self.a_bound, self.a_bound)

                    # noise = (torch.randn_like(action_) * self.policy_noise).clamp(-self.a_bound,self.a_bound)
                    # action_with_noise = ( action_+ noise).clamp(-self.a_bound,self.a_bound)

                # 计算target Q 值
                target_Q1, _ ,target_Q2, _ = self.critic_target(b_s_,action_with_noise,h_s_,h_s_len)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = b_r + self.GAMMA * target_Q

            # 获得当前batch Q estimates
            current_Q1,q1_extracted_memory, current_Q2 ,q2_extracted_memory = self.critic(b_s,b_a,h_s,h_s_len)

            self.loss_info_critic = dict(Q1Vals=current_Q1.detach().cpu().numpy(),
                             Q2Vals=current_Q2.detach().cpu().numpy(),
                             Q1ExtractedMemory=q1_extracted_memory.mean(dim=1).detach().cpu().numpy(),
                             Q2ExtractedMemory=q2_extracted_memory.mean(dim=1).detach().cpu().numpy())


            # 计算critic loss = td - error

            critic_loss = F.mse_loss(current_Q1,target_Q) + F.mse_loss(current_Q2,target_Q)
            self.show_critic_loss = critic_loss
            '''
            if self.use_wandb:
                wandb.log(data={'Q1Vals': loss_info_critic['Q1Vals'],
                            'Q2Vals': loss_info_critic['Q2Vals'],
                            'Q1ExtractedMemory': loss_info_critic['Q1ExtractedMemory'],
                            'Q2ExtractedMemory': loss_info_critic['Q2ExtractedMemory']})
            if self.use_wandb:
                wandb.log({'loss/critic':self.critic_loss})
                
            '''
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 延迟策略更新
            if self.update_cnt % self.policy_target_update_interval == 0:

            # if j % self.policy_target_update_interval == 0:
                action , a_extracted_memory= self.actor(b_s,h_s,h_a,h_s_len)
                Q1_critic , _ =self.critic.Q1(b_s,action,h_s,h_s_len)
                actor_loss = -Q1_critic.mean()
                self.show_actor_loss = actor_loss


                self.loss_info_actor = dict(ActExtractedMemory=a_extracted_memory.mean(dim=1).detach().cpu().numpy())

                '''
                if self.use_wandb:
                    wandb.log(data={'ActExtractedMemory': loss_info_actor['ActExtractedMemory']})
                    wandb.log({'loss/actor': self.actor_loss})
                '''
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
                    # noise = (torch.randn_like(b_a) * self.policy_noise).clamp(-self.a_bound, self.a_bound)
                    action_, _ = self.actor_target(b_s_, h_s_, h_a_, h_s_len)
                    # action_with_noise = (action_ + noise).clamp(-self.a_bound, self.a_bound)
                    sample = torch.distributions.Normal(0., 1.)
                    a_dim = (self.a_dim,)
                    # sample_ = sample.sample(a_dim)
                    noise = torch.clamp(sample.sample(a_dim) * self.eval_noise_scale, -2 * self.eval_noise_scale,
                                        2 * self.eval_noise_scale)
                    action_with_noise = (action_ + noise).clamp(-self.a_bound, self.a_bound)

                # 计算target Q 值
                target_Q1, _, target_Q2, _ = self.critic_target(b_s_, action_with_noise, h_s_, h_s_len)
                target_Q = torch.min(target_Q1, target_Q2)
                # target_Q = torch.tensor(b_r) + self.GAMMA * target_Q
                target_Q = b_r + self.GAMMA * target_Q

            # 获得当前batch Q estimates
            current_Q1, q1_extracted_memory, current_Q2, q2_extracted_memory = self.critic(b_s, b_a, h_s, h_s_len)

            self.loss_info_critic = dict(Q1Vals=current_Q1.detach().cpu().numpy(),
                             Q2Vals=current_Q2.detach().cpu().numpy(),
                             Q1ExtractedMemory=q1_extracted_memory.mean(dim=1).detach().cpu().numpy(),
                             Q2ExtractedMemory=q2_extracted_memory.mean(dim=1).detach().cpu().numpy())
            '''
            if self.use_wandb:
                wandb.log(data={'Q1Vals': loss_info_critic['Q1Vals'],
                            'Q2Vals': loss_info_critic['Q2Vals'],
                            'Q1ExtractedMemory': loss_info_critic['Q1ExtractedMemory'],
                            'Q2ExtractedMemory': loss_info_critic['Q2ExtractedMemory']})
            '''

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
        return self.show_critic_loss,self.show_actor_loss


class Memory(object):
    def __init__(self, capacity, dims,pick_len):
        self.capacity = capacity
        self.pick_len = pick_len + 1
        self.data = np.zeros((capacity, dims))
        self.pointer = 0
        self.exp_rep = ExpRep(capacity, self.pick_len, True,False,False)
        self.sample_rate=0.03
        self.size = 0


    def store_transition(self,h_epi, s, a, r, s_):
        # 在此处将array转为c_int64
        a = self.exp_rep.encoded_actions(action=a)
        self.size = min(self.size+1,self.capacity)
        return self.exp_rep.record(h_epi,s,a,r,s_)


    def sample(self, n, a_dim):
        # assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        # indices = np.random.choice(self.capacity, size=n)
        # return self.data[indices, :]
        return self.exp_rep.get_random_batch_lstm(n,self.sample_rate,a_dim=a_dim)

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