
import warnings
import gym
from Agent import Config
import warnings
import copy
import time
import numpy as np
import pandas as pd
# import tensorflow as tf
from new_calculate import *
from Agent.TD3_LSTM import TD3 as TD3_lstm
import wandb

wandb.init(project="cortex22_anna", name='LSTM_pen', notes="LSTM立杆子")
warnings.filterwarnings('ignore')

io_path = 'io/'
ex_path = io_path + 'enterprise_nnu/'
logs_path = ex_path + 'logs'
session_path = ex_path + 'session'
model_filename = ex_path + 'model'

clustered_devices = None


class enterprise_nnu:
    def __init__(self, config: Config):
        self.enterprise = TD3_lstm(config=config)  # 生成num个mod
        self.epi = None
        self.last_state = None  # 银行家i的银行的上一个state
        self.max_hist_len = config.MAX_HIST_LEN
        self.a_dim = config.action_dim
        self.s_dim = config.state_dim
        self.ptr = self.enterprise.get_pointer()

    def run_enterprise(self, state,new_ep):  # enterprise_mod范围:[1, num]

        # =====准备工作=====#
        if new_ep:
            h_epi = None  # 准备h_epi
            # 初始化 short history buff
            max_hist_len = self.max_hist_len
            if max_hist_len > 0:
                self.state_buff = np.zeros([max_hist_len, self.s_dim])
                self.action_buff = np.zeros([max_hist_len, self.a_dim])
                self.state_buff[0, :] = state
                self.state_buff_len = 0
            else:
                self.state_buff = np.zeros([1, self.s_dim])
                self.action_buff = np.zeros([1, self.a_dim])
                self.state_buff_len = 0
        else:
            h_epi = self.epi
        state = np.array(state)  # state准备就绪

        # =====得到action=====#
        h_epi,action = self.enterprise.choose_action_lstm(h_epi,state,self.state_buff,self.action_buff,self.state_buff_len)

        if new_ep:
            self.epi = copy.deepcopy(h_epi)
        return action

    def env_upd(self, state, action, state_, reward, is_train, is_end=False):  # 这里的h_epi是upd_epi
        # =====准备工作=====#
        state_ = np.array(state_)  # state准备就绪
        self.last_state = state_
        # print(self.scope, ":", self.epi)
        # =====把[state, action, reward, next_state]按序存入h_epi所属部分, 更新epi_map=====#

        if is_train:

            # self.enterprise.episode_feedback(state, action, reward, state_ if is_end else None,is_end)
            # feedback, store replay_buff
            self.epi = self.enterprise.episode_feedback(self.epi, state, action, reward, state_ if is_end else None)

            # Add short history
            max_hist_len = self.max_hist_len
            if self.state_buff_len == max_hist_len:
                self.state_buff[:max_hist_len - 1] = self.state_buff[1:]
                self.action_buff[:max_hist_len - 1] = self.action_buff[1:]
                self.state_buff[max_hist_len - 1] = list(state)
                self.action_buff[max_hist_len - 1] = list(action)
            else:
                self.state_buff[self.state_buff_len + 1 - 1] = list(state)
                self.action_buff[self.state_buff_len + 1 - 1] = list(action)
                self.state_buff_len += 1


            # update, compute loss
            loss = self.enterprise.learn()
            return loss
        return None


def lstm_td3(config = Config,env_name='', seed=0,
             steps_per_epoch=4000, epochs=1000, gamma=0.99,
             batch_size=100,
             max_hist_len=100,
             lra=0.0001,lrc=0.0002,lrd=0.98
             ):
    env = gym.make('Pendulum-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    config = config(
        batch_lstm = batch_size,
        max_hist_len= max_hist_len,
        learning_rate_actor=lra,
        learning_rate_decay=lrd,
        learning_rate_critic=lrc,
        random_seed=seed,
        scope='',
        action_dim=action_dim,
        action_bound=max_action,
    )
    config.set_state_dim(state_dim)
    Agent = enterprise_nnu(config)

    for epoch in range(epochs):
        state = env.reset()
        epo_reward = 0
        done = False
        new_ep = True
        while True:

            action = Agent.run_enterprise(state,new_ep)
            new_ep = False

            state_,reward, done, _= env.step(action)

            Agent.env_upd(state,action,state_,reward,is_train=True,is_end=done)
            epo_reward += reward
            # 移动到下一个状态
            state = state_

            if done:
                break

        wandb.log({"reward":epo_reward})
        print("reward:",epo_reward)

    wandb.finish()
    # 关闭环境
    env.close()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name',type=str,default='')
    parser.add_argument('--seed',type=int , default=184)
    parser.add_argument('--epochs',type=int, default=2000)
    parser.add_argument('--max_hist_len',type=int ,default=4)
    parser.add_argument('--batch_size',type=int,default=15)
    parser.add_argument('--learning_rate_actor',type=float, default=0.0001)
    parser.add_argument('--learning_rate_critic',type=float,default=0.002)
    parser.add_argument('--learning_rate_decay',type=float,default=1.0)
    parser.add_argument('--exp_name', type=str, default='lstm_td3')
    args = parser.parse_args()
    lstm_td3(env_name=args.env_name,seed=args.seed,epochs=args.epochs,max_hist_len=args.max_hist_len,
             batch_size=args.batch_size, lra=args.learning_rate_actor,lrc=args.learning_rate_critic,lrd=args.learning_rate_decay)



