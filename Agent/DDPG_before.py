import tensorflow as tf
import numpy as np
from Agent.Common.ExperienceReplay_DDPG import Experience_Replay as ExpRep
from Agent.cpp_DDPG import  Config
tranLock = False
isPercent = True
class DDPG(object):
    def __init__(self, config: Config, scope: str):
        self.a_dim = config.action_dim
        self.s_dim = config.state_dim
        self.a_bound = config.action_bound
        self.scope = scope
        self.memory = Memory(capacity=config.MEMORY_CAPACITY, dims=2 * self.s_dim + self.a_dim + 1)
        self.LR_A = config.LEARNING_RATE_ACTOR
        self.LR_C = config.LEARNING_RATE_CRITIC
        self.GAMMA = config.REWARD_GAMMA
        self.TAU = config.SOFT_REPLACE_TAU
        self.BATCH_SIZE = config.BATCH_SIZE
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.pointer = 0
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_dim))
        self.episode_temp = {}

        self.S = tf.placeholder(tf.float32, [None, self.s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.var_init = config.VAR_INIT
        self.var_stable = config.VAR_STABLE
        self.var_drop_at = config.VAR_DROP_AT
        self.var_stable_at = config.VAR_STABLE_AT
        self.var_end_at = config.VAR_END_AT

        self.var = self.var_init

        with tf.variable_scope('Actor' + self.scope):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic' + self.scope):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor' + self.scope + '/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor' + self.scope + '/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic' + self.scope + '/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic' + self.scope + '/target')



        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - self.TAU) * t + self.TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + self.GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(self.td_error, var_list=self.ce_params)

        self.a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(self.a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, h_epi,s):
        self.var = self.var_init
        if self.pointer > self.var_stable_at:
            self.var = self.var_stable
        else:
            delta_step = self.pointer - self.var_drop_at
            if delta_step > 0:self.var = self.var_init + delta_step * (self.var_stable-self.var_init)/(self.var_stable_at-self.var_drop_at)
        if is_list(h_epi):
            ep_num = len(h_epi)
            ret_h_epi = [None] * ep_num
            ret_action = [None] * ep_num
            for i in range(len(h_epi)):
                h,a = self.choose_action(h_epi[i],s)
                ret_h_epi[i] = h
                ret_action[i] = a
            return ret_h_epi,ret_action
        if h_epi is None:
            h_epi = self.memory.new_ep()
            self.episode_temp[h_epi] = dict()
        temp = self.episode_temp[h_epi]
        if len(np.array(s).shape) == 1:
            action = self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        else:
            action = self.sess.run(self.a, {self.S: s})
        print("before" + self.scope,action)
        if isPercent:
            action = np.clip(action +np.random.normal([0 for i in range(self.a_dim)], self.var), -self.a_bound, self.a_bound) #百分比
            for i in range(len(action)):
                if action[i][0] > self.a_bound:
                    action[i][0] = action[i][0] % self.a_bound
                if action[i][0] < -self.a_bound:
                    action[i][0] = action[i][0] % -self.a_bound
        else :
            action = np.clip(action + np.random.normal([0 for i in range(self.a_dim)],self.var), 0.001, 100000000) # 固定值
            # action = action + abs(np.random.normal([0 for i in range(self.a_dim)],self.var))

        # action = action + self.noise(self.var)
        print("after" + self.scope, action)

        temp['state'] = s
        temp['action'] = action
        return h_epi,action
    def episode_feedback(self,h_epi,reward,final_state):
        self.pointer += 1
        print("_______________________________________@(@*#(@#*( " ,self.pointer,self.var, "_______________________________24$@A#@$")
        temp = self.episode_temp[h_epi]
        ret_h_epi = self.memory.store_transition(h_epi, temp['state'], temp['action'], reward, final_state)
        return ret_h_epi
    def learn(self):
        if self.pointer < 3000:
            return (0,0)
        # soft target replacement
        self.sess.run(self.soft_replace)
        # indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        # bt = self.memory[indices, :]
        # bs = bt[:, :self.s_dim]
        # ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        # br = bt[:, -self.s_dim - 1: -self.s_dim]
        # bs_ = bt[:, -self.s_dim:]

        b_M = self.memory.sample(self.BATCH_SIZE)
        b_s = b_M[0].reshape(-1, self.s_dim)
        # b_a = b_M[1].reshape(-1,action_dim)/10000-2.
        b_a = np.array(b_M[1]).reshape(-1, self.a_dim)
        b_r = b_M[2].reshape(-1, 1)
        b_s_ = b_M[3].reshape(-1, self.s_dim)
        if (not tranLock) or self.pointer < self.var_end_at:
            self.sess.run(self.atrain, {self.S: b_s})
            self.sess.run(self.ctrain, {self.S: b_s, self.a: b_a, self.R: b_r, self.S_: b_s_})
        loss_a = self.sess.run(self.td_error, {self.S: b_s, self.a: b_a, self.R: b_r, self.S_: b_s_})

        return (loss_a,0)

    def store_transition(self,h_epi, s, a, r, s_):
        # transition = np.hstack((s, a, [r], s_))
        # index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # self.memory[index, :] = transition
        self.pointer += 1
        self.memory.store_transition(h_epi, s, a, r, s_)

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            # x_norm = self.batch_norm_layer(s,training_phase=trainable,scope_bn='batch_norm0',activation=tf.nn.relu)
            # l1 = tf.layers.dense(x_norm, 40, activation=tf.nn.relu, name='l1', trainable=trainable)
            # x_norm1 = self.batch_norm_layer(l1,training_phase=trainable,scope_bn='batch_norm1',activation=tf.nn.relu)
            # l2 = tf.layers.dense(x_norm1, 80, activation=tf.nn.relu, name='l2', trainable=trainable)
            # x_norm2 = self.batch_norm_layer(l2,training_phase=trainable,scope_bn='batch_norm2',activation=tf.nn.relu)
            # l3 = tf.layers.dense(x_norm2, 40, activation=tf.nn.relu, name='l3', trainable=trainable)
            # x_norm3 = self.batch_norm_layer(l3,training_phase=trainable,scope_bn='batch_norm3',activation=tf.nn.relu)
            # l4 = tf.layers.dense(x_norm3, 20, activation=tf.nn.relu, name='l4', trainable=trainable)
            # x_norm4 = self.batch_norm_layer(l4, training_phase=trainable, scope_bn='batch_norm4', activation=tf.nn.relu)
            # l5 = tf.layers.dense(x_norm4, 10, activation=tf.nn.relu, name='l5', trainable=trainable)
            # x_norm5 = self.batch_norm_layer(l5, training_phase=trainable, scope_bn='batch_norm5', activation=tf.nn.relu)
            # a = tf.layers.dense(x_norm5, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            # return tf.multiply(a, self.a_bound, name='scaled_a')
            # x_norm = self.batch_norm_layer(s, training_phase=trainable, scope_bn='batch_norm0', activation=tf.nn.relu)
            l1 = tf.layers.dense(s, 10, activation=tf.nn.relu, name='l1', trainable=trainable)
            x_norm1 = self.batch_norm_layer(l1,training_phase=trainable,scope_bn='batch_norm1',activation=tf.nn.relu )
            # l2 = tf.layers.dense(x_norm1, 80, activation=tf.nn.relu, name='l2', trainable=trainable)
            # x_norm2 = self.batch_norm_layer(l2,training_phase=trainable,scope_bn='batch_norm2',activation=tf.nn.tanh)
            # l3 = tf.layers.dense(x_norm2, 40, activation=tf.nn.relu, name='l3', trainable=trainable)
            # x_norm3 = self.batch_norm_layer(l3,training_phase=trainable,scope_bn='batch_norm3',activation=tf.nn.tanh)
            # l4 = tf.layers.dense(x_norm3, 20, activation=tf.nn.relu, name='l4', trainable=trainable)
            # x_norm4 = self.batch_norm_layer(l4, training_phase=trainable, scope_bn='batch_norm4', activation=tf.nn.tanh)
            # l5 = tf.layers.dense(x_norm4, 10, activation=tf.nn.relu, name='l5', trainable=trainable)
            # x_norm5 = self.batch_norm_layer(l5, training_phase=trainable, scope_bn='batch_norm5', activation=tf.nn.tanh)
            # net = tf.layers.dense(s, 10, activation=tf.nn.relu, name='l1', trainable=trainable)
            if isPercent:
                a = tf.layers.dense(x_norm1, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            else:
                a = tf.layers.dense(x_norm3, self.a_dim, activation=tf.nn.relu, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 10
            n_l2 = 80
            n_l3 = 40
            n_l4 = 20
            n_l5 = 10

            # w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            # w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            # b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            # net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            # x_norm = self.batch_norm_layer(net, training_phase=trainable, scope_bn='batch_norm_c0',
            #                                activation=tf.nn.relu)
            # l1 = tf.layers.dense(x_norm, n_l2, activation=tf.nn.relu, name='l1', trainable=trainable)
            # x_norm1 = self.batch_norm_layer(l1, training_phase=trainable, scope_bn='batch_norm_c1',
            #                                 activation=tf.nn.relu)
            # l2 = tf.layers.dense(x_norm1, n_l3, activation=tf.nn.relu, name='l2', trainable=trainable)
            # x_norm2 = self.batch_norm_layer(l2, training_phase=trainable, scope_bn='batch_norm_c2',
            #                                 activation=tf.nn.relu)
            # l3 = tf.layers.dense(x_norm2, n_l4, activation=tf.nn.relu, name='l3', trainable=trainable)
            # x_norm3 = self.batch_norm_layer(l3, training_phase=trainable, scope_bn='batch_norm_c3',
            #                                 activation=tf.nn.relu)
            # l4 = tf.layers.dense(x_norm3, n_l5, activation=tf.nn.relu, name='l4', trainable=trainable)
            # x_norm4 = self.batch_norm_layer(l4, training_phase=trainable, scope_bn='batch_norm_c4',
            #                                 activation=tf.nn.relu)
            #
            # return tf.layers.dense(x_norm4, 1, activation=tf.nn.relu, trainable=trainable, name='l5')  # Q(s,a)

            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            x_norm = self.batch_norm_layer(net, training_phase=trainable, scope_bn='batch_norm_c0',
                                           activation=tf.nn.relu)
            # l1 = tf.layers.dense(x_norm, n_l2, activation=tf.nn.relu, name='l1', trainable=trainable)
            # x_norm1 = self.batch_norm_layer(l1, training_phase=trainable, scope_bn='batch_norm_c1',
            #                                 activation=tf.nn.relu)
            # l2 = tf.layers.dense(x_norm1, n_l3, activation=tf.nn.relu, name='l2', trainable=trainable)
            # x_norm2 = self.batch_norm_layer(l2, training_phase=trainable, scope_bn='batch_norm_c2',
            #                                 activation=tf.nn.relu)
            # l3 = tf.layers.dense(x_norm2, n_l4, activation=tf.nn.relu, name='l3', trainable=trainable)
            # x_norm3 = self.batch_norm_layer(l3, training_phase=trainable, scope_bn='batch_norm_c3',
            #                                 activation=tf.nn.relu)
            # l4 = tf.layers.dense(x_norm3, n_l5, activation=tf.nn.relu, name='l4', trainable=trainable)
            # x_norm4 = self.batch_norm_layer(l4, training_phase=trainable, scope_bn='batch_norm_c4',
            #                                 activation=tf.nn.relu)
            return tf.layers.dense(x_norm, 1, activation=tf.nn.relu, trainable=trainable, name='l5')  # Q(s,a)

    def batch_norm_layer(self, x, training_phase, scope_bn, activation=None):
        return tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
                                            updates_collections=None, is_training=True, reuse=None,
                                            scope=scope_bn, decay=0.9, epsilon=1e-5)
            # tf.cond(training_phase,
            #    lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
            #                                         updates_collections=None, is_training=True, reuse=None,
            #                                         scope=scope_bn, decay=0.9, epsilon=1e-5),
            #    lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
            #                                         updates_collections=None, is_training=False, reuse=True,
            #                                         scope=scope_bn, decay=0.9, epsilon=1e-5))



class Memory(object):

    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0
        self.exp_rep = ExpRep(capacity, 1, False,True,True)
        self.sample_rate=0.03

    def store_transition(self,h_epi, s, a, r, s_):
        # transition = np.hstack((s, a, [r], s_))
        # index = self.pointer % self.capacity  # replace the old memory with new memory
        # self.data[index, :] = transition
        # self.pointer += 1
        # self.exp_rep.record(h_epi,s,a,r,None)
        s = s[0].reshape(-1)
        return self.exp_rep.record(h_epi,s,a,r,s_)


    def sample(self, n):
        # assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        # indices = np.random.choice(self.capacity, size=n)
        # return self.data[indices, :]
        return self.exp_rep.get_random_batch(n,self.sample_rate)

    def new_ep(self):
        return self.exp_rep.new_episode()


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.02, theta=1, dt=1e-2, x0=None):
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