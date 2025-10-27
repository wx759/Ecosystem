__all__ = ['build', 'do_flow']


from .__import__ import tf
from .__import__ import Network
from .__import__ import Device
from .__import__ import SectionTimer
from .Scopes import *
import tensorflow as tf_root
__tfv__ = tf_root.__version__
if __tfv__ >= '1.14':
    tf = tf_root.compat.v1
    tf.where = tf_root.compat.v2.where
else:
    tf = tf_root

def build(dev:Device) -> Network:
    network = Network()
    serv = dev.host
    config = serv.config
    s_shape = config.s_shape
    batch_size = config.train_batch_size
    pick_len = config.pick_len
    shape0 = [batch_size]
    shape1 = [batch_size, pick_len]
    shape2 = shape1 + s_shape
    build_R_net = (config.R_network_func is not None)
    network.key_tensors = { 'flows': list() }
    with network.dispatch_to('/gpu:0'):
    # with network.dispatch_to('/cpu:0'):
    # with tf.device('/gpu:0') :
        with network.root_scope(ROOT_SCOPE_SERVER(serv.id)):
            with network.scope(GLOABL_SCOPE_BATCH_PIPE):
                src_batch_ticket = tf.Variable(0, dtype = tf.int64, trainable = False, collections = [], name = 'batch_ticket')
                src_seq_len0 = tf.Variable(tf.zeros(shape0, dtype = tf.int64), trainable = False, collections = [], name = 'seq_len0')
                src_seq_len1 = tf.Variable(tf.zeros(shape0, dtype = tf.int64), trainable = False, collections = [], name = 'seq_len1')
                src_act_taken = tf.Variable(tf.zeros(shape1, dtype = tf.int64), trainable = False, collections = [], name = 'act_taken')
                src_state0 = tf.Variable(tf.zeros(shape2, dtype = tf.float32), trainable = False, collections = [], name = 'state0')
                src_state1 = tf.Variable(tf.zeros(shape2, dtype = tf.float32), trainable = False, collections = [], name = 'state1')
                if not build_R_net:
                    src_reward = tf.Variable(tf.zeros(shape1, dtype = tf.float32), trainable = False, collections = [], name = 'reward')
    with network.dispatch_to(dev.tf_code):
        with network.root_scope(ROOT_SCOPE_DEVICE(dev.id)) as root_scope:
            with network.scope(GLOABL_SCOPE_BATCH_PIPE):
                batch_ticket = tf.Variable(0, dtype = tf.int64, trainable = False, collections = [], name = 'batch_ticket')
                network.key_tensors['flows'].append(tf.assign(batch_ticket, src_batch_ticket, name = 'flow_batch_ticket'))
                seq_len0 = tf.Variable(tf.zeros(shape0, dtype = tf.int64), trainable = False, collections = [], name = 'seq_len0')
                network.key_tensors['flows'].append(tf.assign(seq_len0, src_seq_len0, name = 'flow_seq_len0'))
                seq_len1 = tf.Variable(tf.zeros(shape0, dtype = tf.int64), trainable = False, collections = [], name = 'seq_len1')
                network.key_tensors['flows'].append(tf.assign(seq_len1, src_seq_len1, name = 'flow_seq_len1'))
                act_taken = tf.Variable(tf.zeros(shape1, dtype = tf.int64), trainable = False, collections = [], name = 'act_taken')
                network.key_tensors['flows'].append(tf.assign(act_taken, src_act_taken, name = 'flow_act_taken'))
                state0 = tf.Variable(tf.zeros(shape2, dtype = tf.float32), trainable = False, collections = [], name = 'state0')
                network.key_tensors['flows'].append(tf.assign(state0, src_state0, name = 'flow_state0'))
                state1 = tf.Variable(tf.zeros(shape2, dtype = tf.float32), trainable = False, collections = [], name = 'state1')
                network.key_tensors['flows'].append(tf.assign(state1, src_state1, name = 'flow_state1'))
                if not build_R_net:
                    reward = tf.Variable(tf.zeros(shape1, dtype = tf.float32), trainable = False, collections = [], name = 'reward')
                    network.key_tensors['flows'].append(tf.assign(reward, src_reward, name = 'flow_reward'))
        network.session_new(target = serv.server.target, initialize = False, inter_op_threads = serv.tf_inter_op_threads)
    return network


def do_flow(network:Network, timer:SectionTimer):
    timer.enter_timing_section()
    flow_tensors = network.key_tensors['flows']
    try: network._sess[0].run(flow_tensors)
    except: pass
    timer.exit_timing_section()