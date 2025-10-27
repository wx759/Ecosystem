import tensorflow as tf
import tensorflow_probability as tfp
from tfdeterminism import patch
patch()
import numpy as np
import random
import math
import copy
import os
from Agent.Common.ExperienceReplay_DDPG import Experience_Replay as ExpRep
from real_System.MADDPG.MADDPGConfig import  MADDPGConfig
from Agent.RuningMeanStd import RunningMeanStd
from Agent.RuningMeanStd import TfRunningMeanStd

tranLock = True
isPercent = True
agent_num = 4
k=3
state_offset = 0
no = {
    'production1_business':0,
    'consumption1_business':1,
    'production2_business':2,
    'consumption2_business':3,
}
class MADDPG(object):
    def __init__(self, config: MADDPGConfig):
        self.a_dim = config.action_dim
        self.s_dim = config.state_dim
        self.a_bound = config.action_bound
        self.scope = config.scope
        self.memory = []
        for i in range(k):
            self.memory.append(Memory(capacity=config.MEMORY_CAPACITY, dims= agent_num * (2 * self.s_dim + self.a_dim) + 1))
        self.current_agent_no = 0
        self.LR_A = config.LEARNING_RATE_ACTOR
        self.LR_C = config.LEARNING_RATE_CRITIC
        self.LR_A_STABLE = config.LEARNING_RATE_ACTOR_STABLE
        self.LR_C_STABLE = config.LEARNING_RATE_CRITIC_STABLE
        self.LR_DECAY = config.LEARNING_RATE_DECAY
        self.LR_DECAY_TIME = config.LEARNING_RATE_DECAY_TIME
        self.GAMMA = config.REWARD_GAMMA
        self.TAU = config.SOFT_REPLACE_TAU
        self.BATCH_SIZE = config.BATCH_SIZE
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
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

        if self.is_delay:
            self.policy_target_update_interval = config.ACTOR_UPDATE_DELAY_TIMES # 策略网络更新频率
        else:
            self.policy_target_update_interval = 1

        if self.is_smooth:
            self.eval_noise_scale = config.SMOOTH_NOISE  # 评估动作噪声缩放
        else:
            self.eval_noise_scale = 0.0  # 评估动作噪声缩放



        self.set_global_seed(config.random_seed)

        self.S = tf.placeholder(tf.float32, [None, self.s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
        self.other_S = tf.placeholder(tf.float32, [None, (agent_num - 1) * self.s_dim - (agent_num - 1) * state_offset], 'other_s')
        self.other_S_ = tf.placeholder(tf.float32, [None, (agent_num - 1) * self.s_dim - (agent_num - 1) * state_offset], 'other_s_')
        self.other_A = tf.placeholder(tf.float32,[None,(agent_num - 1) * self.a_dim],'other_a')
        self.other_A_ = tf.placeholder(tf.float32,[None,(agent_num - 1) * self.a_dim],'other_a_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.bn_is_train = tf.placeholder_with_default(False,(),'bn_is_train')
        self.c_learning_rate = tf.placeholder_with_default(self.LR_C,(),'LR_C')
        self.a_learning_rate = tf.placeholder_with_default(self.LR_A,(),'LR_A')

        self.var_init = config.VAR_INIT
        self.var_stable = config.VAR_STABLE
        self.var_drop_at = config.VAR_DROP_AT
        self.var_stable_at = config.VAR_STABLE_AT
        self.var_end_at = config.VAR_END_AT

        self.var = self.var_init

        with tf.variable_scope('Actor' + self.scope):
            self.a,self.net_show = self._build_a(self.S, scope='eval', trainable=True)
            self.a_,_ = self._build_a(self.S_, scope='target', trainable=False)
            if self.is_smooth:
                sample = tf.distributions.Normal(loc=0., scale=1.)
                noise = tf.clip_by_value(sample.sample(self.a_dim) * self.eval_noise_scale, -2 * self.eval_noise_scale,
                                         2 * self.eval_noise_scale)
                noise_a_ = tf.clip_by_value(self.a_ + noise,-self.a_bound,self.a_bound)
            else:
                noise_a_ = self.a_

        with tf.variable_scope('Critic' + self.scope):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q1,_ = self._build_c(self.S, self.a,self.other_S,self.other_A, scope='eval1', trainable=True)
            q1_,self.net_show_critic1 = self._build_c(self.S_,noise_a_,self.other_S_,self.other_A_, scope='target1', trainable=False)
            if self.is_double:
                q2, self.net_show_critic2 = self._build_c(self.S, self.a,self.other_S,self.other_A, scope='eval2', trainable=True)
                q2_, _ = self._build_c(self.S_,noise_a_,self.other_S_,self.other_A_, scope='target2', trainable=False)
        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor' + self.scope + '/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor' + self.scope + '/target')
        self.ce_params1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic' + self.scope + '/eval1')
        self.ct_params1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic' + self.scope + '/target1')
        if self.is_double:
            self.ce_params2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic' + self.scope + '/eval2')
            self.ct_params2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic' + self.scope + '/target2')


        # target net replacement
        if self.is_double:
            self.soft_replace = [tf.assign(t, (1 - self.TAU) * t + self.TAU * e)
                                 for t, e in zip(self.at_params + self.ct_params1 + self.ct_params2,
                                                 self.ae_params + self.ce_params1 + self.ce_params2)]
            self.hard_replace = [tf.assign(t, e)
                                 for t, e in zip(self.at_params + self.ct_params1 + self.ct_params2,
                                                 self.ae_params + self.ce_params1 + self.ce_params2)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.TAU) * t + self.TAU * e)
                                 for t, e in zip(self.at_params + self.ct_params1,
                                                 self.ae_params + self.ce_params1)]
            self.hard_replace = [tf.assign(t, e)
                                 for t, e in zip(self.at_params + self.ct_params1,
                                                 self.ae_params + self.ce_params1)]

        # self.soft_replace = [tf.assign(t, (1 - self.TAU) * t + self.TAU * e)
                             # for t, e in zip(self.at_params + self.ct_params1, self.ae_params + self.ce_params1)]
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # self.update_ops_eval = tf.get_collection(tf.GraphKeys.UPDATE_OPS)[-20:-14] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)[-8:-6] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)[-4:-2]
        # self.update_ops_target = tf.get_collection(tf.GraphKeys.UPDATE_OPS)[-14:-8] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)[-6:-4] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)[-2:]
        self.update_ops_eval = tf.get_collection(tf.GraphKeys.UPDATE_OPS)[-24:-18]  + tf.get_collection(tf.GraphKeys.UPDATE_OPS)[-12:-6]
        self.update_ops_target = tf.get_collection(tf.GraphKeys.UPDATE_OPS)[-18:-12]  + tf.get_collection(tf.GraphKeys.UPDATE_OPS)[-6:]
        # self.update_ops_eval = tf.get_collection(tf.GraphKeys.UPDATE_OPS)[-12:-6]
        # self.update_ops_target = tf.get_collection(tf.GraphKeys.UPDATE_OPS)[-6:]
        self.update_ops_eval = []
        self.update_ops_target = []
        with tf.control_dependencies(self.update_ops_target):
            if self.is_double:
                self.q_min = tf.minimum(q1_, q2_)
            else:
                self.q_min = q1_
            self.q_target = self.R + self.GAMMA * self.q_min

        # in the feed_dic for the td_error, the self.a should change to actions in memory
            self.td_error1 = tf.losses.mean_squared_error(labels=self.q_target, predictions=self.q1)
            self.ctrain1 = tf.train.AdamOptimizer(self.c_learning_rate).minimize(self.td_error1, var_list=self.ce_params1)
            if self.is_double:
                self.td_error2 = tf.losses.mean_squared_error(labels=self.q_target, predictions=q2)
                self.ctrain2 = tf.train.AdamOptimizer(self.c_learning_rate).minimize(self.td_error2, var_list=self.ce_params2)
        with tf.control_dependencies(self.update_ops_eval):
            self.a_loss = - tf.reduce_mean(self.q1)    # maximize the q
            self.atrain = tf.train.AdamOptimizer(self.a_learning_rate).minimize(self.a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.hard_replace)


    def choose_action(self, h_epi,s):
        # s = self.translate(s)
        c = np.array(s)[np.newaxis,:]
        self.rms.update(c)
        if self.is_rms:
            self.toshow['归一化前输入']=copy.deepcopy(s)
            s = (s-self.rms.mean)/(self.rms.var + 1e-5)
        self.var = self.var_init
        if self.pointer > self.var_stable_at:
            self.var = self.var_stable
        else:
            delta_step = self.pointer - self.var_drop_at
            if delta_step > 0:self.var = self.var_init + delta_step * (self.var_stable-self.var_init)/(self.var_stable_at-self.var_drop_at)

        if h_epi is None:
            h_epi = self.memory[self.current_agent_no].new_ep()
            self.episode_temp[h_epi] = dict()
        temp = self.episode_temp[h_epi]
        to_run = [self.a]
        keys = []
        for key in self.net_show:
            to_run.append(self.net_show[key])
            keys.append(key)
        # for key in self.net_show_critic1:
        #     to_run.append(self.net_show_critic1[key])
        #     keys.append(key)
        # if self.is_double:
        #     for key in self.net_show_critic2:
        #         to_run.append(self.net_show_critic2[key])
        #         keys.append(key)
        self.toshow['归一化后输入'] = s
        self.toshow['rms_mean'] = self.rms.mean
        self.toshow['rms_var'] = self.rms.var

        list = self.sess.run(to_run, feed_dict={self.S: s[np.newaxis, :]})
        # toshow = {'l1': showl1, 'l1bn': showl1bn, 'l1output': l1,
        #           'l2': showl2, 'l2bn': showl2bn, 'l2output': l2,
        #           'a': showa, 'abn': showabn,
        #           }
        # {'cl1': showl1, 'cl1bn': showl1bn, 'cl1output': l1,
        #  'critic': showa, 'criticbn': showabn
        #  }
        action = list[0][0]
        for i in range(len(keys)):
            self.toshow[keys[i]] = list[i+1][0]

        if isPercent:
            action = action + np.random.normal([0 for i in range(self.a_dim)], self.var)
            # action = np.clip(action, -self.a_bound, self.a_bound)
            for i in range(len(action)):
               if action[i] > self.a_bound:
                   action[i] = action[i] % self.a_bound
               if action[i] < -self.a_bound:
                   action[i] = action[i] % -self.a_bound
        else:
            action = np.clip(action + np.random.normal([0 for i in range(self.a_dim)],self.var), 0.001, 100000000) # 固定值
            # action = action + abs(np.random.normal([0 for i in range(self.a_dim)],self.var))

        # action = action + self.noise(self.var)
        # print("after" + self.scope, action)

        # temp['state'] = s
        # temp['action'] = action
        return h_epi,action

    def check_show(self):
        return self.toshow

    def episode_feedback(self,h_epi,state,other_state,action,other_action,reward,final_state,other_final_state):
        # 很明显这是无论是逻辑上还是它的诞生过程都是一个bug
        # 是我曾经的一个尝试，尝试宣告失败时将其关入了注释的牢笼
        # 而之后调试一直不顺利，在某一次意外操作中
        # 不小心将这个bug从注释中释放出来了
        # 神奇的是，模型竟然收敛了
        # 重新注释掉这个bug，模型又跑不动了
        # 我不知道为什么
        # 我只知道我这条命都是他给的
        # 别叫他bug，叫他爸爸
        # if final_state is not None:
        #     final_state = self.translate(final_state)
        self.pointer += 1
        if self.pointer%100 == 0:
            print(self.mark())
        # temp = self.episode_temp[h_epi]
        x = np.concatenate([state,other_state],axis=0)
        action = action + other_action
        final_x = final_state
        if final_state is not None:
            final_x = np.concatenate([final_state,other_final_state],axis=0)
            final_x = self.translate(final_x)
        ret_h_epi = self.memory[self.current_agent_no].store_transition(h_epi, x, action, reward, final_x)
        return ret_h_epi

    def mark(self):
        return "_______________________________________@(@*#(@#*( "  + str(self.pointer) +" " +str(self.var) + " " + str(self.show_lar_a)+ "_______________________________24$@A#@$"

    def get_batch(self):
        if self.pointer < 3000:
            return None
        self.b_M = self.memory[self.current_agent_no].sample(self.BATCH_SIZE)
        return copy.deepcopy(self.b_M)

    def get_action_to_enviroment(self,other_agent_batch):
        # episode_feedback中采用将自身state存放在前而另一智能体state存放在后的储存方式，所以这里从后面取
        self_no = no[self.scope]
        begin = self_no
        end = self_no + 1
        other_agent_s = other_agent_batch[0].reshape(-1,agent_num * self.s_dim)[:,begin * self.s_dim:end * self.s_dim]
        other_agent_s_ = other_agent_batch[3].reshape(-1,agent_num * self.s_dim)[:,begin * self.s_dim:end * self.s_dim]
        to_other_a,to_other_a_ = self.sess.run([self.a,self.a_],{self.S:other_agent_s,self.S_:other_agent_s_})
        return to_other_a,to_other_a_


    def learn(self,other_action,other_action_):
        if self.pointer < 3000:
            return 0
        # soft target replacement
        self.update_cnt += 1
        other_agent_action = other_action[0]
        other_agent_action_ = other_action_[0]
        for i in range(1,len(other_action)):
            other_agent_action = np.concatenate([other_agent_action,other_action[i]],axis=1)
            other_agent_action_ = np.concatenate([other_agent_action_,other_action_[i]],axis=1)


        self_no = no[self.scope]
        begin = self_no
        end = self_no + 1

        b_M = self.b_M
        if self.is_rms:
            b_s_rm = (b_M[0]-self.rms.mean)/(self.rms.var+1e-5)
            b_s__rm = (b_M[3]-self.rms.mean)/(self.rms.var+1e-5)
        else:
            b_s_rm = b_M[0]
            b_s__rm = b_M[3]

        b_s = b_s_rm.reshape(-1, agent_num * self.s_dim)
        b_self_s = b_s[:,begin * self.s_dim:end * self.s_dim]
        b_other_s =np.concatenate([b_s[:,0:begin * self.s_dim],b_s[:,end * self.s_dim:]],axis=1)

        b_s_ = b_s__rm.reshape(-1, agent_num * self.s_dim)
        b_self_s_ = b_s_[:, begin * self.s_dim:end * self.s_dim]
        b_other_s_ = np.concatenate([b_s_[:,0:begin * self.s_dim],b_s_[:,end * self.s_dim:]],axis=1)
        # b_self_s_ = b_s_[:, :self.s_dim]
        # b_other_s_ = b_s_[:, self.s_dim:2 * self.s_dim - state_offset]

        b_a = np.array(b_M[1]).reshape(-1, agent_num * self.a_dim)
        b_self_a = b_a[:,begin * self.a_dim:end * self.a_dim]
        b_other_a = np.concatenate([b_a[:,0:begin * self.a_dim],b_a[:,end * self.a_dim:]],axis=1)

        # debug区
        # b_other_s = b_self_s
        # b_other_s_ = b_other_s
        # b_other_a = b_self_a
        # other_agent_action = self.sess.run(self.a,{self.S:b_self_s})
        # other_agent_action_ = self.sess.run(self.a_,{self.S_:b_self_s_})



        b_r = b_M[2].reshape(-1, 1)
        lr_a = max(self.LR_A_STABLE,self.LR_A * np.power(self.LR_DECAY,((self.pointer-3000)/self.LR_DECAY_TIME)) )
        lr_c = max(self.LR_C_STABLE,self.LR_C * np.power(self.LR_DECAY,((self.pointer-3000)/self.LR_DECAY_TIME)) )
        self.show_lar_a = lr_a
        if (not tranLock) or self.pointer < self.var_end_at:
            _,loss_c1,q_min,q_target,l1,q1,l2,c = self.sess.run([self.ctrain1,self.td_error1,self.q_min,self.q_target,self.net_show_critic1['cl1'],self.q1,self.net_show_critic1['cl2'],self.net_show_critic1['cout']], {self.S: b_self_s, self.a: b_self_a,self.S_: b_self_s_, self.R: b_r,
                                                                                                              self.other_S:b_other_s,self.other_S_:b_other_s_,
                                                                                                              self.other_A:b_other_a,self.other_A_:other_agent_action_,
                                                                                                              self.bn_is_train:True,self.c_learning_rate:lr_c})
            if self.is_double:
                _,loss_c2 = self.sess.run([self.ctrain2,self.td_error2], {self.S: b_self_s, self.a: b_self_a,self.S_: b_self_s_, self.R: b_r,
                                                                          self.other_S:b_other_s,self.other_S_:b_other_s_,
                                                                          self.other_A:b_other_a,self.other_A_:other_agent_action_,
                                                                          self.bn_is_train:True,self.c_learning_rate:lr_c})
            if self.update_cnt % self.policy_target_update_interval == 0:
                self.sess.run(self.atrain, {self.S: b_self_s,self.other_S:b_other_s,self.other_A:other_agent_action,
                                            self.bn_is_train:True,self.a_learning_rate:lr_a})
                self.sess.run(self.soft_replace)
            self.toshow['qmin']=[np.array(q_min).reshape(-1).mean()]
            self.toshow['qtarget']=[np.array(q_target).reshape(-1).mean()]
        else:
            loss_c1 = self.sess.run(self.td_error1,  {self.S: b_self_s, self.a: b_self_a,self.S_: b_self_s_, self.R: b_r,
                                                      self.other_S:b_other_s,self.other_S_:b_other_s_,
                                                      self.other_A:b_other_a,self.other_A_:other_agent_action_,
                                                      self.bn_is_train:True,self.c_learning_rate:lr_c})
            if self.is_double:
                loss_c2 = self.sess.run(self.td_error2,  {self.S: b_self_s, self.a: b_self_a,self.S_: b_self_s_, self.R: b_r,
                                                      self.other_S:b_other_s,self.other_S_:b_other_s_,
                                                      self.other_A:b_other_a,self.other_A_:other_agent_action_,
                                                      self.bn_is_train:True,self.c_learning_rate:lr_c})

        if self.is_double:
            return min(loss_c1,loss_c2)
        else:
            return loss_c1

    def store_transition(self,h_epi, s, a, r, s_):
        # transition = np.hstack((s, a, [r], s_))
        # index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # self.memory[index, :] = transition
        self.pointer += 1
        self.memory[self.current_agent_no].store_transition(h_epi, s, a, r, s_)

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            # l1 = tf.layers.dense(s, 20, activation=tf.nn.tanh, name='l1', trainable=trainable)
            # # x_norm1 = self.batch_norm_layer(l1,training_phase=trainable,scope_bn='batch_norm1',activation=tf.nn.leaky_relu )
            # l2 = tf.layers.dense(l1, 5, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)
            # # x_norm2 = self.batch_norm_layer(l2,training_phase=trainable,scope_bn='batch_norm2',activation=tf.nn.tanh)
            # a = tf.layers.dense(l2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)

            l1,showl1,showl1bn = self.dense_tanh_toshow(s, 128, training_phase=trainable, scope='layer1')
            # l1 = tf.multiply(l1,5e-2,name='scaled_l1')
            l2,showl2,showl2bn = self.dense_leaky_relu_toshow(l1, 32, training_phase=trainable, scope='layer2')
            # l2 = tf.multiply(l2,5e-2,name='scaled_l2')
            a,showa,showabn = self.dense_tanh_toshow(l2, self.a_dim, training_phase=trainable, scope='a')
            toshow = {
                # 'l1':showl1,'l1bn':showl1bn,'l1output':l1,
                #       'l2':showl2,'l2bn':showl2bn,'l2output':l2,
                      'a':showa,'abn':showabn,'aout':a
                      }
            return tf.multiply(a, self.a_bound, name='scaled_a'),toshow

    def _build_c(self, s, a,other_s,other_a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 128
            n_l2 = 32
            # n_l3 = 10
            # n_l4 = 20
            # n_l5 = 10

            new_s = tf.concat([s,other_s],axis=1)
            new_a = tf.concat([a,other_a],axis=1)

            w1_s = tf.get_variable('w1_s', [self.s_dim * agent_num , n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim * agent_num, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)

            net = tf.matmul(new_s, w1_s) + tf.matmul(new_a, w1_a) + b1
            bn1 = self.batch_norm_layer(net,training_phase=trainable,scope='layer1bn')

            l1 = tf.nn.leaky_relu(net)
            # l1 = tf.multiply(l1,5e-2,name='scaled_l1')

            # l1,showl1,showl1bn = self.dense_tanh_toshow(net,n_l1,training_phase=trainable,scope='layer1')
            l2,showl2,showl2bn = self.dense_leaky_relu_toshow(l1,n_l2,training_phase=trainable,scope='layer2')
            # l2 = tf.multiply(l2,5e-2,name='scaled_l2')

            # output,showout,showputbn = self.dense_leaky_relu_toshow(l1,1,training_phase=trainable,scope='output')
            # output,_,_ = self.dense_leaky_relu_toshow(l2,1,training_phase=trainable,scope='output')
            # output,show_output,show_output_bn = self.dense(l2,1,training_phase=trainable,scope='output')
            output = self.dense(l2,1,training_phase=trainable,scope='output')

            toshow = {
                'cl1':net,'cl2':l2,'cl1out':l1,
                # 'cl2':showl2,'cl2bn':showl2bn,'cl2out':l2,
                'cout':output
                      }
            return output,toshow
            # return tf.layers.dense(l1, 1, trainable=trainable,activation=tf.nn.leaky_relu, name='l5'),toshow  # Q(s,a)

    def translate(self,data):
        for d in range(len(data)):
            data[d] = data[d]/1000
        return np.array(data)
    def fully_connected_dense(self, input, output_size, scope, training_phase=True):
        return tf.contrib.layers.fully_connected(input, output_size,
                                                 activation_fn=None,
                                                 scope=scope,
                                                 trainable=training_phase)

    def dense(self, input, output_size, training_phase, scope):
        with tf.variable_scope(scope):
            h1 = self.fully_connected_dense(input=input, output_size=output_size,
                                            scope='dense', training_phase=training_phase)

            return h1

    def dense_tanh(self, input, output_size, training_phase, scope):
        with tf.variable_scope(scope):
            h1 = self.fully_connected_dense(input=input, output_size=output_size,
                                            scope='dense', training_phase=training_phase)
            return tf.nn.tanh(h1, 'tanh')

    def dense_leaky_relu_toshow(self, input, output_size, training_phase, scope):
        with tf.variable_scope(scope):
            bn = self.batch_norm_layer(input=input, training_phase=training_phase,
                                       scope=scope + 'bn')
            h1 = self.fully_connected_dense(input=input, output_size=output_size,
                                            scope='dense', training_phase=training_phase)


            return tf.nn.leaky_relu(h1, name='leaky_relu'),h1,h1

    def dense_tanh_toshow(self, input, output_size, training_phase, scope):
        with tf.variable_scope(scope):
            bn = self.batch_norm_layer(input=input, training_phase=training_phase,
                                       scope=scope + 'bn')
            h1 = self.fully_connected_dense(input=input, output_size=output_size,
                                            scope='dense', training_phase=training_phase)

            return tf.nn.tanh(h1, 'tanh'),h1,h1

    def dense_batch_leaky_relu(self, input, output_size, training_phase, scope):
        with tf.variable_scope(scope):
            bn = self.batch_norm_layer(input=input, training_phase=training_phase,
                                       scope='bn')
            h1 = self.fully_connected_dense(input=bn, output_size=output_size,
                                            scope='dense', training_phase=training_phase)
            return tf.nn.leaky_relu(h1, name='leaky_relu')

    def dense_batch_tanh(self, input, output_size, training_phase, scope):
        with tf.variable_scope(scope):
            h1 = self.fully_connected_dense(input=input, output_size=output_size,
                                            scope='dense', training_phase=training_phase)
            bn = self.batch_norm_layer(input=h1, training_phase=training_phase,
                                       scope=scope + 'bn')
            return tf.nn.tanh(bn, 'tanh')

    def batch_norm_layer(self, input, training_phase, scope):
        with tf.variable_scope(scope):
            return tf.contrib.layers.batch_norm(input, center=True, scale=True,
                                            is_training=self.bn_is_train,
                                            scope=scope,decay=0.9,epsilon=1e-5)

    def set_global_seed(self,seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        tf.set_random_seed(seed)
        np.random.seed(seed)

        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

class Memory(object):

    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0
        self.exp_rep = ExpRep(capacity, 1, False,True,True)
        self.sample_rate=0.03

    def store_transition(self,h_epi, s, a, r, s_):

        return self.exp_rep.record(h_epi,s,a,r,s_)


    def sample(self, n):
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