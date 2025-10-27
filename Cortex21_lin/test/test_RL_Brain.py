import numpy as np
import tensorflow as tf
import  matplotlib.pyplot as plt
import pandas as pd
from Cortex.Common.ExperienceReplay import Experience_Replay as ExpRep
import random
seed = random.randint(1,10000)
np.random.seed(seed)
tf.set_random_seed(seed)

class DeepQNet:
    def __init__(self,
                 action_space_n,
                 n_feature,
                 learning_rate=0.01,#学习率
                 reward_decay=0.9,#奖励衰减
                 e_greedy=0.9,
                 replace_target_iter=300,#更新目标网络的循环次数
                 memory_size=500,#经验池
                 batch_size=32,
                 e_greedy_increment=None,
                 doubleDQN=False,
                 duelingDQN=False,
                 sess=None,
                 output_graph=False,
                 ):
        self.n_action=action_space_n
        self.n_feature=n_feature
        self.lr=learning_rate
        self.gamma=reward_decay
        self.epsilon_max=e_greedy
        self.replace_target_iter=replace_target_iter
        self.memory_size=memory_size
        self.batch_size=batch_size
        self.doubleDQN=doubleDQN
        self.duelingDQN=duelingDQN
        self.e_greedy_increment=e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.exp_rep = ExpRep(memory_size, 1, False)
        self.sample_rate=0.03

        self.learning_step_count=0 #学习步数
        self.memory=np.zeros((self.memory_size,self.n_feature*2+2)) #[state(n_feature),action,reward,next_state(n_feature)]

        self._build_net() #建立网络
        target_net_params = tf.get_collection('target_net_params')
        eval_net_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t,e) for (t,e) in zip(target_net_params,eval_net_params)] #返回值应为eval_net_params
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess=sess
        self.cost_his = []
        self.reward_his = []
        self.q_his = []
    def _build_net(self):
        def build_layers(state, collections_name, n_layer1, w_initializer, b_initializer):
            with tf.variable_scope('layer1'):
                w1=tf.get_variable('w1',[self.n_feature,n_layer1],initializer=w_initializer,collections=collections_name)
                b1=tf.get_variable('b1',[1,n_layer1],initializer=b_initializer,collections=collections_name)
                l1=tf.nn.relu(tf.matmul(state,w1)+b1)
            if self.duelingDQN:
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2',[n_layer1,1],initializer=w_initializer,collections=collections_name)
                    b2 = tf.get_variable('b2',[1,1],initializer=b_initializer,collections=collections_name)
                    self.V = tf.matmul(l1,w2) + b2
                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2',[n_layer1,self.n_action],initializer=w_initializer,collections=collections_name)
                    b2 = tf.get_variable('b2',[1,self.n_action],initializer=b_initializer,collections=collections_name)
                    self.A = tf.matmul(l1,w2) + b2

                with tf.variable_scope('layer2'):
                    out = self.V + (self.A - tf.reduce_mean(self.A,axis=1,keep_dims=True))

            else:
                with tf.variable_scope('layer2'):
                    w2 = tf.get_variable('w2',[n_layer1,self.n_action],initializer=w_initializer,collections=collections_name)
                    b2 = tf.get_variable('b2',[1,self.n_action],initializer=b_initializer,collections=collections_name)
                    # self.q_eval = tf.matmul(l1,w2) + b2 #n_action * 1
                    out = tf.matmul(l1,w2) + b2
            return out

        self.state = tf.placeholder(tf.float32, [None, self.n_feature], name='state')  # 环境状态
        self.q_target = tf.placeholder(tf.float32, [None, self.n_action], name='Q_target')  # 神经网络输出
        with tf.variable_scope('eval_net'):
            collections_name = ['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
            n_layer1 = 30
            w_initializer = tf.random_normal_initializer(0.,0.3)
            b_initializer = tf.constant_initializer(0.1)
            self.q_eval = build_layers(self.state,collections_name,n_layer1,w_initializer,b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval)) #1/n * E(q_target - q_eval)^2

        with tf.variable_scope('train'):
            # self.train_op= tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize((self.loss))

        self.next_state = tf.placeholder(tf.float32,[None,self.n_feature],name='next_state')
        with tf.variable_scope('target_net'):
            collections_name = ['target_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
            # with tf.variable_scope('layer1'):
            #     w1 = tf.get_variable('w1', [self.n_feature, n_layer1], initializer=w_initializer, collections=collections_name)
            #     b1 = tf.get_variable('b1', [1, n_layer1], initializer=b_initializer, collections=collections_name)
            #     l1 = tf.nn.relu(tf.matmul(self.next_state, w1) + b1)
            #
            #     # second layer. collections is used later when assign to target net
            # with tf.variable_scope('layer2'):
            #     w2 = tf.get_variable('w2', [n_layer1, self.n_action], initializer=w_initializer, collections=collections_name)
            #     b2 = tf.get_variable('b2', [1, self.n_action], initializer=b_initializer, collections=collections_name)
            #     self.q_next = tf.matmul(l1, w2) + b2
            self.q_next = build_layers(self.next_state,collections_name,n_layer1,w_initializer,b_initializer)

    def store_transition(self,h_epi,state,action,reward,next_state):
        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((state,[action,reward],next_state))
        index = self.memory_counter % self.memory_size
        self.memory[index , :] = transition
        self.memory_counter+=1

    def choose_action(self,observation):
        observation = observation[np.newaxis,:]

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval,feed_dict={self.state: observation}) #e+eval与state有关
            action = np.argmax(actions_value)
            self.q_his.append(np.max(actions_value))
        else:
            action = np.random.randint(0,self.n_action)
        return action

    def learn(self):
        if self.learning_step_count % self.replace_target_iter == 0 :
            self.sess.run(self.replace_target_op)
            # if self.learning_step_count % 1000 == 0:
                # print('\n target_params_repleaced\n')
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,size=self.batch_size)
        batch_memory = self.memory[sample_index,:]
        # q_next = self.sess.run(self.q_next,feed_dist={self.next_state:batch_memory[:,-self.n_feature:]})
        # q_eval = self.sess.run(self.q_eval,feed_dist={self.state:batch_memory[:,:self.n_feature]})
        # q_next, q_eval = self.sess.run(
        #     [self.q_next, self.q_eval],
        #     feed_dict={
        #         self.next_state: batch_memory[:, -self.n_feature:],  # fixed params
        #         self.state: batch_memory[:, :self.n_feature],  # newest params
        #     })
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.next_state: batch_memory[:, -self.n_feature:],  # next observation
                       self.state: batch_memory[:, -self.n_feature:]})  # next observation
        q_eval = self.sess.run(self.q_eval, {self.state: batch_memory[:, :self.n_feature]})
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size,dtype=np.int32)
        eval_act_index = batch_memory[:,self.n_feature].astype(int) #batch_memory[:,self.n_feature] = batch_memory[:,action]
        reward = batch_memory[:,self.n_feature + 1]
        if self.doubleDQN:
            max_act4next = np.argmax(q_eval4next,axis=1)
            selected_q_next = q_next[batch_index,max_act4next]
        else:
            selected_q_next = np.max(q_next,axis=1)

        q_target[batch_index,eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self.train_op, self.loss],
                                     feed_dict={self.state: batch_memory[:, :self.n_feature],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon= self.epsilon + self.e_greedy_increment if self.epsilon <  self.epsilon_max else self.epsilon_max
        self.learning_step_count+=1
    def finish(self,r):
        self.reward_his.append(r)
    def clear_finish(self):
        self.reward_his=[]
    def plot_cost(self):
        # plt.plot(np.arange(len(self.reward_his)),self.reward_his)
        # plt.ylabel('reward')
        # plt.plot(np.arange(len(self.cost_his)),self.cs)
        # plt.ylabel('epsilon')
        plt.plot(np.arange(len(self.q_his)),self.q_his)
        plt.ylabel('Q')
        plt.xlabel('training steps')
        plt.show()
        print(self.reward_his[-100:].mean)

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            layer_n=20
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.layer_n=layer_n

        self.ep_obs, self.ep_as, self.ep_rs = [], [], [] #该回合的环境集合、动作集合、奖励集合

        self._build_net()
        self.reward_his=[]
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=self.layer_n,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # layer2 = tf.layers.dense(
        #     inputs=layer,
        #     units=10,
        #     activation=tf.nn.tanh,  # tanh activation
        #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
        #     bias_initializer=tf.constant_initializer(0.1),
        #     name='fcn'
        # )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
    def finish(self,r):
        self.reward_his.append(r)
    def clear_finish(self):
        self.reward_his=[]




class Actor(object):
    def __init__(self,
                 sess,
                 n_features,
                 n_actions=None, #nactions与action_bound冲突 一个离散一个连续
                 action_bound=None,
                 lr=0.001):
        self.sess=sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        if n_actions is None:
            self.a = tf.placeholder(tf.float32, None, name="act")
        else:
            self.a = tf.placeholder(tf.int32, None, name="act")
        self.td_error = tf.placeholder(tf.float32,None,name="td_error")
        self.n_actions = n_actions
        self.action_bound = action_bound
        self.lr = lr

        self.__build_net__()
    def __build_net__(self):
        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=30,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )
            global_step = tf.Variable(0, trainable=False)
            if self.n_actions is None:
                mu = tf.layers.dense(
                    inputs=l1,
                    units=1,
                    activation=tf.nn.tanh,
                    kernel_initializer=tf.random_normal_initializer(0.,.1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='mu'
                )

                sigma = tf.layers.dense(
                    inputs=l1,
                    units=1,
                    activation=tf.nn.softplus,
                    kernel_initializer=tf.random_normal_initializer(0.,.1),
                    bias_initializer=tf.constant_initializer(1.),
                    name='sigma'
                )
                self.mu,self.sigma = tf.squeeze(mu*2),tf.squeeze(sigma+0.1)
                self.normal_dist = tf.distributions.Normal(self.mu,self.sigma)
                self.action = tf.clip_by_value(self.normal_dist.sample(1),self.action_bound[0],self.action_bound[1])
            else:
                self.acts_prob = tf.layers.dense(
                    inputs = l1,
                    units = self.n_actions,
                    activation = tf.nn.softmax,
                    kernel_initializer = tf.random_normal_initializer(mean=0,stddev=0.1),
                    bias_initializer = tf.constant_initializer(0.1),
                    name = 'acts_prob'
                )
            with tf.variable_scope('exp_v'):
                if self.n_actions is None:
                    log_prob = self.normal_dist.log_prob(self.a)
                    self.exp_v = log_prob * self.td_error
                    self.exp_v += 0.01*self.normal_dist.entropy()
                else:
                    log_prob = tf.log(self.acts_prob[0, self.a])
                    self.exp_v  = tf.reduce_mean(log_prob * self.td_error)

            with tf.variable_scope('train'):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v,global_step)

    def learn(self,s,a,td):
        s = s[np.newaxis,:]
        feed_dict = {self.s:s,self.a:a,self.td_error:td}
        _,exp_v = self.sess.run([self.train_op,self.exp_v],feed_dict=feed_dict)
        return exp_v
    def choose_action(self,s):
        s = s[np.newaxis,:]
        if self.n_actions is None:
            return self.sess.run(self.action,{self.s:s})
        else:
            probs = self.sess.run(self.acts_prob,feed_dict={self.s:s})
            return np.random.choice(np.arange(probs.shape[1]),p=probs.ravel())

class Critic(object):
    def __init__(self,sess,n_features,n_actions=None,gamma=0.9,lr=0.01):
        self.sess=sess

        self.s = tf.placeholder(tf.float32,[1,n_features],name='state')
        self.v_ = tf.placeholder(tf.float32,[1,1],name='v_next')
        self.r = tf.placeholder(tf.float32,None,name='r')
        self.lr=lr
        self.n_actions=n_actions
        self.gamma = gamma
        self.__build_net__()

    def __build_net__(self):
        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs = self.s,
                units = 30,
                activation = tf.nn.relu,
                kernel_initializer = tf.random_normal_initializer(0,0.1),
                bias_initializer = tf.constant_initializer(0.1),
                name = 'l1'
            )

        self.v = tf.layers.dense(
            inputs = l1,
            units = 1,
            activation = None,
            kernel_initializer = tf.random_normal_initializer(0,0.1),
            bias_initializer = tf.constant_initializer(0.1),
            name='V'
        )

        with tf.variable_scope('squared_TD_error'):
            if self.n_actions is None:
                self.td_error = tf.reduce_mean(self.r + self.gamma * self.v_ - self.v)
                self.loss = tf.square(self.td_error)
            else:
                self.td_error = self.r + self.gamma * self.v_ -  self.v
                self.loss = tf.square(self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    def learn(self,s,r,s_):
        s,s_ = s[np.newaxis,:],s_[np.newaxis,:]

        v_ = self.sess.run(self.v,feed_dict = {self.s:s_})

        td_error,_ = self.sess.run([self.td_error,self.train_op],feed_dict={self.s:s,self.v_:v_,self.r:r})

        return td_error


