import keras
import tensorflow as tf
#import tensorflow.keras import layers
import tensorflow_probability as tfp
from tfdeterminism import patch
patch()
import numpy as np
import random
import math
import copy
import os
from Agent.Common.ExperienceReplay_DDPG import Experience_Replay as ExpRep
from . import  Config
from Agent.RuningMeanStd import RunningMeanStd
from Agent.RuningMeanStd import TfRunningMeanStd

tranLock = True
isPercent = True
class DDPG(object):
    def __init__(self, config: Config):
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
        #LSTM参数
        self.is_lstm = config.IS_LSTM
        self.pre_lstm_hid_sizes = (128,)
        self.lstm_hid_sizes = (128,)
        self.after_lstm_hid_size = (128,)
        self.cur_feature_hid_sizes = (128,)
        self.post_comb_hid_sizes = (128,)
        self.hist_with_past_act = False

        if self.is_delay:
            self.policy_target_update_interval = config.ACTOR_UPDATE_DELAY_TIMES # 策略网络更新频率
        else:
            self.policy_target_update_interval = 1

        if self.is_smooth:
            self.eval_noise_scale = config.SMOOTH_NOISE  # 评估动作噪声缩放
        else:
            self.eval_noise_scale = 0.0  # 评估动作噪声缩放



        self.set_global_seed(config.random_seed)
        # [None,s_dim] 行数不限，列数为s_dim
        self.S = tf.placeholder(tf.float32, [None, self.s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
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

        # create actor (current , target )
        with tf.variable_scope('Actor' + self.scope):
            self.a,self.net_show = self._build_a(self.S, scope='eval', trainable=True)
            a_,_ = self._build_a(self.S_, scope='target', trainable=False)
            if self.is_smooth:
                sample = tf.distributions.Normal(loc=0., scale=1.) #(01正态分布)
                noise = tf.clip_by_value(sample.sample(self.a_dim) * self.eval_noise_scale, -2 * self.eval_noise_scale,
                                         2 * self.eval_noise_scale)
                noise_a_ = tf.clip_by_value(a_ + noise,-self.a_bound,self.a_bound)
            else:
                noise_a_ = a_

        # create critic (current , target )
        with tf.variable_scope('Critic' + self.scope):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q1,self.net_show_critic1 = self._build_c(self.S, self.a, scope='eval1', trainable=True)
            q1_,_ = self._build_c(self.S_,noise_a_, scope='target1', trainable=False)
            if self.is_double:
                q2, self.net_show_critic2 = self._build_c(self.S, self.a, scope='eval2', trainable=True)
                q2_, _ = self._build_c(self.S_, noise_a_, scope='target2', trainable=False)
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

        # soft replace 软更新
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

        # 计算td_error

        # in the feed_dic for the td_error, the self.a should change to actions in memory
            self.td_error1 = tf.losses.mean_squared_error(labels=self.q_target, predictions=q1)
            self.ctrain1 = tf.train.AdamOptimizer(self.c_learning_rate).minimize(self.td_error1, var_list=self.ce_params1)
            if self.is_double:
                self.td_error2 = tf.losses.mean_squared_error(labels=self.q_target, predictions=q2)
                self.ctrain2 = tf.train.AdamOptimizer(self.c_learning_rate).minimize(self.td_error2, var_list=self.ce_params2)
        with tf.control_dependencies(self.update_ops_eval):
            self.a_loss = - tf.reduce_mean(q1)    # maximize the q
            self.atrain = tf.train.AdamOptimizer(self.a_learning_rate).minimize(self.a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.hard_replace)

# 进行动作决策
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
            h_epi = self.memory.new_ep()
            self.episode_temp[h_epi] = dict()
        temp = self.episode_temp[h_epi]
        to_run = [self.a]
        keys = []
        for key in self.net_show:
            to_run.append(self.net_show[key])
            keys.append(key)
        for key in self.net_show_critic1:
            to_run.append(self.net_show_critic1[key])
            keys.append(key)
        if self.is_double:
            for key in self.net_show_critic2:
                to_run.append(self.net_show_critic2[key])
                keys.append(key)
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
# 每个训练周期结束时进行反馈，调整智能体的行为，根据反馈信号更新Actor和Critic
    def episode_feedback(self,h_epi,state,action,reward,final_state):
        # state = self.translate(state)
        if final_state is not None:
            final_state = np.zeros(len(final_state))
        self.pointer += 1
        if self.pointer%100 == 0:
            print(self.mark())
        # temp = self.episode_temp[h_epi]
        ret_h_epi = self.memory.store_transition(h_epi, state, action, reward, final_state)
        return ret_h_epi

    def mark(self):
        return "_______________________________________@(@*#(@#*( "  + str(self.pointer) +" " +str(self.var) + " " + str(self.show_lar_a)+ "_______________________________24$@A#@$"

    # 训练
    def learn(self):

        # 判断当前训练步数是否达到一定阈值（3000）以开始训练，如果没有达到则不开始训练
        if self.pointer < 3000:
            return 0
        # 在更新AC网络之前通过update_cnt计数器更新目标网络的参数
        # 在把eval网络的参数赋值给target网络时使用soft replacement 可以避免不收敛的问题
        # soft target replacement
        self.update_cnt += 1

        # indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        # bt = self.memory[indices, :]
        # bs = bt[:, :self.s_dim]
        # ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        # br = bt[:, -self.s_dim - 1: -self.s_dim]
        # bs_ = bt[:, -self.s_dim:]
        if self.pointer>8000 and self.pointer%100 == 0:
            a=1

        # replay buffer中采随机采样的样本数据
        b_M = self.memory.sample(self.BATCH_SIZE)
        # 数据处理（归一化） ，b_s_rm,b_s__rm 当前状态和下一个状态的处理后的观测数据
        if self.is_rms:
            b_s_rm = (b_M[0]-self.rms.mean)/(self.rms.var+1e-5)
            b_s__rm = (b_M[3]-self.rms.mean)/(self.rms.var+1e-5)
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
        lr_a = max(self.LR_A_STABLE,self.LR_A * np.power(self.LR_DECAY,((self.pointer-3000)/self.LR_DECAY_TIME)) )
        lr_c = max(self.LR_C_STABLE,self.LR_C * np.power(self.LR_DECAY,((self.pointer-3000)/self.LR_DECAY_TIME)) )
        self.show_lar_a = lr_a

#--------------train--------------------------------------
        # 根据条件判断选择不同训练方式
        # 运行self.ctrain1操作，更新值函数网络，并计算损失函数self.td_error1、q_min和q_target。
        # 如果使用双Q网络（is_double为真），
        # 运行self.ctrain2操作，更新第二个值函数网络，并计算第二个损失函数loss_c2。
        if (not tranLock) or self.pointer < self.var_end_at:
            _,loss_c1,q_min,q_target = self.sess.run([self.ctrain1,self.td_error1,self.q_min,self.q_target], {self.S: b_s, self.a: b_a, self.R: b_r, self.S_: b_s_,self.bn_is_train:True,self.c_learning_rate:lr_c})
            if self.is_double:
                _,loss_c2 = self.sess.run([self.ctrain2,self.td_error2], {self.S: b_s, self.a: b_a, self.R: b_r, self.S_: b_s_,self.bn_is_train:True,self.c_learning_rate:lr_c})
            if self.update_cnt % self.policy_target_update_interval == 0:
                self.sess.run(self.atrain, {self.S: b_s,self.bn_is_train:True,self.a_learning_rate:lr_a})
                self.sess.run(self.soft_replace)
            self.toshow['qmin']=[np.array(q_min).reshape(-1).mean()]
            self.toshow['qtarget']=[np.array(q_target).reshape(-1).mean()]

        # 只计算损失函数loss_c1和loss_c2，但不进行网络参数的更新（不运行对应的self.ctrain操作）
        else:
            loss_c1 = self.sess.run(self.td_error1, {self.S: b_s, self.a: b_a, self.R: b_r, self.S_: b_s_})
            if self.is_double:
                loss_c2 = self.sess.run(self.td_error2, {self.S: b_s, self.a: b_a, self.R: b_r, self.S_: b_s_})

        # 如果使用了双Q网络，则返回loss_c1和loss_c2中较小的一个作为损失函数，否则返回loss_c1。
        if self.is_double:
            return min(loss_c1,loss_c2)
        else:
            return loss_c1
# 将经验存储到缓冲池（memory）中
    # 参数：时间步数（或者称作episode的索引）、当前状态、选择的动作、获得的奖励以及下一个状态
    def store_transition(self,h_epi, s, a, r, s_):
        # transition = np.hstack((s, a, [r], s_))
        # index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # self.memory[index, :] = transition
        self.pointer += 1  # 存储了一条新经验
        self.memory.store_transition(h_epi, s, a, r, s_) # 将当前状态、动作、奖励和下一个状态作为一条完整的经验（transition）存储到经验缓冲池中

# 构建策略网络Actor
    # 参数： 状态，命名空间，是否可训练
    def _build_a(self, s, scope, trainable):
        # 在给定的命名空间scope下
        with tf.variable_scope(scope):
            # l1 = tf.layers.dense(s, 20, activation=tf.nn.tanh, name='l1', trainable=trainable)
            # # x_norm1 = self.batch_norm_layer(l1,training_phase=trainable,scope_bn='batch_norm1',activation=tf.nn.leaky_relu )
            # l2 = tf.layers.dense(l1, 5, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)
            # # x_norm2 = self.batch_norm_layer(l2,training_phase=trainable,scope_bn='batch_norm2',activation=tf.nn.tanh)
            # a = tf.layers.dense(l2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)




            # 添加全连接层和输出层
            '''
            dense1 = tf.layers.dense(lstm_output, 20, activation=tf.nn.tanh, name='dense1', trainable=trainable)
            dense2 = tf.layers.dense(dense1, 5, activation=tf.nn.leaky_relu, name='dense2', trainable=trainable)
            a = tf.layers.dense(dense2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            '''


            # 构建具有128个神经元的全连接层，激活函数为tanh，输出为l1

            l1,showl1,showl1bn = self.dense_tanh_toshow(s, 128, training_phase=trainable, scope='layer1')
            # l1 = tf.multiply(l1,5e-2,name='scaled_l1')
            # 构建具有32个神经元的全连接层，激活函数为leaky ReLU，输出为l2
            l2,showl2,showl2bn = self.dense_leaky_relu_toshow(l1, 32, training_phase=trainable, scope='layer2')
            # l2 = tf.multiply(l2,5e-2,name='scaled_l2')
            # 建立一个输出层，神经元数量为动作空间的维度self.a_dim，激活函数为tanh
            a,showa,showabn = self.dense_tanh_toshow(l2, self.a_dim, training_phase=trainable, scope='a')
            toshow = {
                # 'l1':showl1,'l1bn':showl1bn,'l1output':l1,
                #       'l2':showl2,'l2bn':showl2bn,'l2output':l2,
                      'a':showa,'abn':showabn,'aout':a
                      }
            # 将输出action限制在动作空间范围内，进行缩放，a_bound是a的上界
            return tf.multiply(a, self.a_bound, name='scaled_a'),toshow

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 128
            n_l2 = 32
            # n_l3 = 10
            # n_l4 = 20
            # n_l5 = 10


            # LSTM
            n_lstm_units = 64  # lstm单元数量
            # 创建lstm层
            lstm_cell = tf.keras.layers.LSTMCell(units=n_lstm_units)
            lstm_layer = tf.keras.layers.RNN(lstm_cell)
            # 将状态s传递给lstm层
            inputs = tf.concat([s,a],axis=-1)
            lstm_output = lstm_layer(inputs)
            new_inputs = tf.concat([inputs,lstm_output],axis=-1)




            # 权重矩阵w1_s，w1_a，偏置项b1
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)

            # 中间层的输出：net
            net = tf.matmul(new_inputs,w1_s) + tf.matmul(a,w1_a) + b1
            # net = tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1
            # 对net进行归一化
            bn1 = self.batch_norm_layer(net,training_phase=trainable,scope='layer1bn')

            # 使用 leaky ReLU 激活函数对 net 进行非线性处理得到 l1。
            l1 = tf.nn.leaky_relu(net)
            # l1 = tf.multiply(l1,5e-2,name='scaled_l1')

            # 并通过调用 self.dense_leaky_relu_toshow 函数构建具有 32 个神经元的其他全连接层 l2
            # l1,showl1,showl1bn = self.dense_tanh_toshow(net,n_l1,training_phase=trainable,scope='layer1')
            l2,showl2,showl2bn = self.dense_leaky_relu_toshow(l1,n_l2,training_phase=trainable,scope='layer2')
            # l2 = tf.multiply(l2,5e-2,name='scaled_l2')

            # output,showout,showputbn = self.dense_leaky_relu_toshow(l1,1,training_phase=trainable,scope='output')
            # output,_,_ = self.dense_leaky_relu_toshow(l2,1,training_phase=trainable,scope='output')
            # output,show_output,show_output_bn = self.dense(l2,1,training_phase=trainable,scope='output')
            # 输出层
            output = self.dense(l2,1,training_phase=trainable,scope='output')

            toshow = {
                # 'cl1':net,'cl1bn':bn1,'cl1out':l1,
                # 'cl2':showl2,'cl2bn':showl2bn,'cl2out':l2,
                'c':output,'cbn':output,'cout':output
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
        os.environ['PYTHONHASHSEED'] = str(seed) # 设置python的hash随机种子，确保一些内部操作在不同的运行中也会产生相同的结果，在涉及一些哈希散列操作的情况下很有用
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