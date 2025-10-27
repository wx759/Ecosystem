import  gym
import  numpy as np
import  tensorflow as tf
from test.test_RL_Brain import  DeepQNet
import matplotlib.pyplot as plt
import math
env = gym.make('CartPole-v0')
env = env.unwrapped
sess = tf.Session()
with tf.variable_scope('DQN'):
    DQN=DeepQNet(action_space_n=env.action_space.n,
            n_feature=env.observation_space.shape[0],
            learning_rate=0.01,
            e_greedy=0.95,
            memory_size=10000,
            e_greedy_increment=0.001,
            doubleDQN=False
            )
with tf.variable_scope('Dueling_DQN'):
    duelingDQN = DeepQNet(action_space_n=env.action_space.n,
                         n_feature=env.observation_space.shape[0],
                         learning_rate=0.01,
                         e_greedy=0.95,
                         memory_size=10000,
                         e_greedy_increment=0.001,
                         duelingDQN=True
                         )
sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    reward_his=[]
    list = np.zeros(100)
    for i_episode in range(300):
        print(i_episode)

        observation = env.reset()
        ep_r = 0
        i = 0
        if list.mean() > 195:
            print(i_episode, " finished")
            break
        while True:
            # env.render()
            action = RL.choose_action(observation)
            observation_, reward, done, info = env.step(action)


            if done:
                reward=-200
            # print(done)
            RL.store_transition(observation, action, reward, observation_)
            ep_r += 1

            if total_steps > 1000:
                RL.learn()

            i += 1
            if done or ep_r >= 199:
                list = np.hstack((list[1:], [math.ceil(ep_r)]))
                RL.finish(ep_r)
                print('episode:', i_episode, ' ep_r:', ep_r, 'mean', list.mean())
                reward_his.append(list.mean())
                break

            observation = observation_
            total_steps += 1
    return reward_his
q_natural = train(DQN)
q_deuling = train((duelingDQN))

plt.plot(np.array(q_natural),c='r',label='natural')
plt.plot(np.array(q_deuling),c='b',label='deuling')
plt.legend(loc='best')
plt.ylabel('reward')
plt.xlabel('training steps')
plt.grid()
plt.show()
# print(q_natural.mean())
# print(q_deuling.mean())