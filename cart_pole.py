import gym
import warnings
import time
from Cortex import *
from pandas import DataFrame


warnings.filterwarnings('ignore')

io_path = 'io/'
ex_path = io_path + 'cart_pole/'
logs_path = ex_path + 'logs'
session_path = ex_path + 'session'
model_filename = ex_path + 'model'

cluster_config = None

_DEFAULT_X_BOUND = 2
_DEFAULT_THETA_BOUND = 0.4

start = time.clock()

def _custom_reward(state, x_bound = _DEFAULT_X_BOUND, theta_bound = _DEFAULT_THETA_BOUND):
    x, x_dot, theta, theta_dot = state
    fall = (abs(x) > x_bound or abs(theta) > theta_bound)
    s_x = x_bound - abs(x)
    if s_x < 0: s_x = 0
    s_theta = theta_bound - abs(theta)
    if s_theta < 0: s_theta = 0
    reward = s_x * s_theta
    return [reward, fall]

ep_num_max = 800
step_num_succ = 3000
learning_rate = 0.05
train_batch_size = 4000
train_seq_len = 8
allow_short_seq = False
valid_sample_rate = 0.02
dropout_rate = 0.02
q_decay = 0.95
upd_shadow_period = 200
experience_size = 10000
eps_stable_at_trained_step = 5000
feeding_buffer = 0
mod_num = 1

env = gym.make('CartPole-v1')
action_num = env.action_space.n
state_size = env.observation_space.shape[0]





def Q_network_func(network:Network, state_name:str, q_name:str, has_shadow:bool):
    network.add_layer_full_conn('FC1', state_name, 'H1', state_size * 8, act_func = ActivationFunc.leaky_relu,
                                       dropout_rate_name = 'dropout_rate', has_shadow = has_shadow)
    network.add_layer_full_conn('FC2', 'H1', 'H2', state_size * 4, act_func = ActivationFunc.leaky_relu, has_shadow = has_shadow)
    # network.add_layer_full_conn('LSTM1', 'H2', 'H3', state_size * 4, has_shadow=has_shadow)
    network.add_layer_lstm('LSTM1', 'H2', 'H3', state_size * 4, has_shadow = has_shadow)
    network.add_layer_full_conn('FC3', 'H3', 'A', action_num, has_shadow = has_shadow)
    network.add_layer_full_conn('FC4', 'H3', 'V', 1, has_shadow = has_shadow)
    network.add_layer_duel_q('Duel_Q', 'V', 'A', q_name, has_shadow = has_shadow)

config = ActorDQN.Config(
    experience_size = experience_size,
    pick_len = train_seq_len,
    allow_short_seq = allow_short_seq,
    train_batch_size = train_batch_size,
    valid_sample_rate = valid_sample_rate,
    Q_network_func = Q_network_func,
    R_network_func = None,
    s_shape = [state_size],
    q_decay = q_decay,
    upd_shadow_period = upd_shadow_period,
    optimizer = Optimizer.Adam,
    double_dqn = True,
    pick_selector_class = PickSelector.greedy_bin_heap
)
actor = ActorDQN(config, mod_num, cluster_config = cluster_config)

# t = actor.print_graphs_thread(logs_path, True)
# t.join()

if actor.model_import(model_filename) == []:
    print('empty')
else:
    print('not empty')


for e in range(ep_num_max):
    state = env.reset()
    h_epi = None
    fall = False
    step = 0

    while not fall:
        env.render()
        h_epi, action = actor.episode_act(h_epi, state)

        state = env.step(action)[0]
        step += 1
        reward, fall = _custom_reward(state)
        if step >= step_num_succ: fall = True
        h_epi = actor.episode_feedback(h_epi, reward, state if fall else None)

        train_ret = actor.model_train(learning_rate, dropout_rate_dict = {'dropout_rate': dropout_rate})
        # print('train_ret:', train_ret)
        if isinstance(train_ret, tuple) and train_ret[0] == ActorDQN.FLAG_SHADOW_UPD:
            print('Shadow network updated.')
    actor.model_export(model_filename)
    print('Episode #', e, ' lasts for ', step, ' steps.')



    if step >= step_num_succ:
        print('Succeed!')
        break
# label = []
# for i in range(len(trend)):
#     label.append(i)
# d2 = dict(zip(label,trend))
# d={}
# d.update(d2)
# df = DataFrame(
#     data = trend
# )
# df.to_csv(
#     "c://Users//liwenjian//Desktop//Cortex190909//竖杆子lstm.csv",
#     index = False,
#     encoding = 'utf-8_sig'
# )
# end = time.clock()
# print('\n运行时间：', (end-start)/60, '分钟')

actor.close()
env.close()
