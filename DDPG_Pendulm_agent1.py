"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time
from Agent.Common.ExperienceReplay_DDPG import Experience_Replay as ExpRep
from Agent.DDPG import DDPG
from Agent.Config import Config


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v1'

###############################  DDPG  ####################################
# class Memory(object):
#
#     def __init__(self, capacity, dims):
#         self.capacity = capacity
#         self.data = np.zeros((capacity, dims))
#         self.pointer = 0
#         self.exp_rep = ExpRep(capacity, 1, False,True,True)
#         self.sample_rate=0.03
#
#     def store_transition(self,h_epi, s, a, r, s_):
#
#         return self.exp_rep.record(h_epi,s,a,r,s_)
#
#
#     def sample(self, n):
#         # assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
#         # indices = np.random.choice(self.capacity, size=n)
#         # return self.data[indices, :]
#         return self.exp_rep.get_random_batch(n,self.sample_rate)
#
#     def new_ep(self):
#         return self.exp_rep.new_episode()
#
# class DDPG(object):
#     def __init__(self, a_dim, s_dim, a_bound,):
#         self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
#         self.pointer = 0
#         self.sess = tf.Session()
#         self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
#         self.memory = Memory(capacity=MEMORY_CAPACITY, dims=2 * self.s_dim + self.a_dim + 1)
#
#         self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
#         self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
#         self.R = tf.placeholder(tf.float32, [None, 1], 'r')
#
#         with tf.variable_scope('Actorccac'):
#             self.a = self._build_a(self.S, scope='eval', trainable=True)
#             a_ = self._build_a(self.S_, scope='target', trainable=False)
#         with tf.variable_scope('Criticccac'):
#             # assign self.a = a in memory when calculating q for td_error,
#             # otherwise the self.a is from Actor when updating Actor
#             q = self._build_c(self.S, self.a, scope='eval', trainable=True)
#             q_ = self._build_c(self.S_, a_, scope='target', trainable=False)
#
#         # networks parameters
#         self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actorccac/eval')
#         self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actorccac/target')
#         self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Criticccac/eval')
#         self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Criticccac/target')
#
#         # target net replacement
#         self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
#                              for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]
#
#         q_target = self.R + GAMMA * q_
#         # in the feed_dic for the td_error, the self.a should change to actions in memory
#         td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
#         self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
#
#         a_loss = - tf.reduce_mean(q)    # maximize the q
#         self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
#
#         self.sess.run(tf.global_variables_initializer())
#
#     def choose_action(self,h_epi, s):
#         if h_epi is None:
#             h_epi = self.memory.new_ep()
#         return h_epi,self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
#
#     def learn(self):
#         # soft target replacement
#         self.sess.run(self.soft_replace)
#
#         # indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
#         # bt = self.memory[indices, :]
#         # bs = bt[:, :self.s_dim]
#         # ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
#         # br = bt[:, -self.s_dim - 1: -self.s_dim]
#         # bs_ = bt[:, -self.s_dim:]
#         b_M = self.memory.sample(BATCH_SIZE)
#         bs = b_M[0].reshape(-1, self.s_dim)
#         # b_a = b_M[1].reshape(-1,action_dim)/10000-2.
#         ba = np.array(b_M[1]).reshape(-1, self.a_dim)
#         br = b_M[2].reshape(-1, 1)
#         bs_ = b_M[3].reshape(-1, self.s_dim)
#
#         self.sess.run(self.atrain, {self.S: bs})
#         self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
#
#     def store_transition(self,h_epi, s, a, r, s_):
#         # transition = np.hstack((s, a, [r], s_))
#         # index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
#         # self.memory[index, :] = transition
#         self.pointer += 1
#         self.memory.store_transition(h_epi, s, a, r, s_)
#
#     def _build_a(self, s, scope, trainable):
#         with tf.variable_scope(scope):
#             net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
#             a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
#             return tf.multiply(a, self.a_bound, name='scaled_a')
#
#     def _build_c(self, s, a, scope, trainable):
#         with tf.variable_scope(scope):
#             n_l1 = 30
#             w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
#             w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
#             b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
#             net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
#             return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
ddpg_config =Config(
    scope='test',
    state_dim=3,
    action_dim=1,
    action_bound=2.0,
    memory_capacity=10000,
    batch_size=32,
    reward_gamma=0.9,
    soft_replace_tau=0.01,
    var_drop_at=32,
    var_init=0,
    var_stable=0,
    var_stable_at=20000,
    # var_end_at=70000,
    learning_rate_actor=0.001,
    learning_rate_critic=0.002,
    # state_dim=,

)


s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(ddpg_config)

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    h_epi = None
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        h_epi,a = ddpg.choose_action(h_epi,s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(h_epi,s, a, r / 10, s_ if j == MAX_EP_STEPS-1 else None)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break
print('Running time: ', time.time() - t1)