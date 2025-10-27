__all__ = ['build', 'do_flow']


from .__import__ import tf
from .__import__ import Network
from .__import__ import Server
from .Scopes import *


def build(serv:Server) -> Network:
    network = Network()
    config = serv.config
    s_shape = config.s_shape
    batch_size = config.train_batch_size
    pick_len = config.pick_len
    shape0 = [batch_size]
    shape1 = [batch_size, pick_len]
    shape2 = shape1 + s_shape
    build_R_net = (config.R_network_func is not None)
    network.key_tensors = { 'feeds': dict(), 'flows':list() }
    with network.dispatch_to('/cpu:0'):
        with network.root_scope(ROOT_SCOPE_SERVER(serv.id)) as root_scope:
            with network.scope(GLOABL_SCOPE_BATCH_PIPE):
                network.key_tensors['feeds']['batch_ticket'] = src_batch_ticket = \
                    tf.placeholder(dtype = tf.int64, name = 'src_batch_ticket')
                batch_ticket = tf.Variable(0, dtype = tf.int64, trainable = False, collections = [], name = 'batch_ticket')
                network.key_tensors['flows'].append(tf.assign(batch_ticket, src_batch_ticket, name = 'flow_batch_ticket'))
                network.key_tensors['feeds']['seq_len0'] = src_seq_len0 = \
                    tf.placeholder(shape = shape0, dtype = tf.int64, name = 'src_seq_len0')
                seq_len0 = tf.Variable(tf.zeros(shape0, dtype = tf.int64), trainable = False, collections = [], name = 'seq_len0')
                network.key_tensors['flows'].append(tf.assign(seq_len0, src_seq_len0, name = 'flow_seq_len0'))
                network.key_tensors['feeds']['seq_len1'] = src_seq_len1 = \
                    tf.placeholder(shape = shape0, dtype = tf.int64, name = 'src_seq_len1')
                seq_len1 = tf.Variable(tf.zeros(shape0, dtype = tf.int64), trainable = False, collections = [], name = 'seq_len1')
                network.key_tensors['flows'].append(tf.assign(seq_len1, src_seq_len1, name = 'flow_seq_len1'))
                network.key_tensors['feeds']['action'] = src_act_taken = \
                    tf.placeholder(shape = shape1, dtype = tf.int64, name = 'src_act_taken')
                act_taken = tf.Variable(tf.zeros(shape1, dtype = tf.int64), trainable = False, collections = [], name = 'act_taken')
                network.key_tensors['flows'].append(tf.assign(act_taken, src_act_taken, name = 'flow_act_taken'))
                network.key_tensors['feeds']['state0'] = src_state0 = \
                    tf.placeholder(shape = shape2, dtype = tf.float32, name = 'src_state0')
                state0 = tf.Variable(tf.zeros(shape2, dtype = tf.float32), trainable = False, collections = [], name = 'state0')
                network.key_tensors['flows'].append(tf.assign(state0, src_state0, name = 'flow_state0'))
                network.key_tensors['feeds']['state1'] = src_state1 = \
                    tf.placeholder(shape = shape2, dtype = tf.float32, name = 'src_state1')
                state1 = tf.Variable(tf.zeros(shape2, dtype = tf.float32), trainable = False, collections = [], name = 'state1')
                network.key_tensors['flows'].append(tf.assign(state1, src_state1, name = 'flow_state1'))
                if not build_R_net:
                    network.key_tensors['feeds']['reward'] = src_reward = \
                        tf.placeholder(shape = shape1, dtype = tf.float32, name = 'src_reward')
                    reward = tf.Variable(tf.zeros(shape1, dtype = tf.float32), trainable = False, collections = [], name = 'reward')
                    network.key_tensors['flows'].append(tf.assign(reward, src_reward, name = 'flow_reward'))
        network.session_new(target = serv.server.target, initialize = False, inter_op_threads = serv.tf_inter_op_threads)
    return network


def do_flow(network:Network, batch:list):
    state0, action, reward, state1, seq_len0, seq_len1, batch_ticket = batch
    feed_dict = {
        network.key_tensors['feeds']['state0']: state0,
        network.key_tensors['feeds']['state1']: state1,
        network.key_tensors['feeds']['seq_len0']: seq_len0,
        network.key_tensors['feeds']['seq_len1']: seq_len1,
        network.key_tensors['feeds']['action']: action,
        network.key_tensors['feeds']['batch_ticket']: batch_ticket
    }
    if 'reward' in network.key_tensors['feeds'].keys():
        feed_dict[network.key_tensors['feeds']['reward']] = reward
    flow_tensors = network.key_tensors['flows']
    try: network._sess[0].run(flow_tensors, feed_dict = feed_dict)
    except: pass
