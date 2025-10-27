__all__ = ['build', 'do_flow']


from .__import__ import tf
from .__import__ import Network
from .__import__ import Model
from .__import__ import SectionTimer
from .__import__ import IDLE_TIME_SHORT
from .__import__ import FLAG_NORMAL
from .__import__ import FLAG_SHADOW_UPD
from .Scopes import *

from time import sleep


def build(mod:Model) -> Network:
    network = Network()
    dev = mod.host
    serv = dev.host
    config = serv.config
    s_shape = config.s_shape
    batch_size = config.train_batch_size
    pick_len = config.pick_len
    shape0 = [batch_size]
    shape1 = [batch_size, pick_len]
    shape2 = shape1 + s_shape
    build_R_net = (config.R_network_func is not None)
    q_name = 'reserved_node_Q'
    shadow_q_name = 'reserved_node_shadow_Q'
    s_name = 'state'
    r_name = 'reserved_node_R'
    network.key_tensors = { 'flows': list() }
    with network.dispatch_to(dev.tf_code):
        with network.root_scope(ROOT_SCOPE_DEVICE(dev.id)) as dev_root_scope:
            with network.scope(GLOABL_SCOPE_BATCH_PIPE):
                src_batch_ticket = tf.Variable(0, dtype = tf.int64, trainable = False, collections = [], name = 'batch_ticket')
                src_seq_len0 = tf.Variable(tf.zeros(shape0, dtype = tf.int64), trainable = False, collections = [], name = 'seq_len0')
                src_seq_len1 = tf.Variable(tf.zeros(shape0, dtype = tf.int64), trainable = False, collections = [], name = 'seq_len1')
                src_act_taken = tf.Variable(tf.zeros(shape1, dtype = tf.int64), trainable = False, collections = [], name = 'act_taken')
                src_state0 = tf.Variable(tf.zeros(shape2, dtype = tf.float32), trainable = False, collections = [], name = 'state0')
                src_state1 = tf.Variable(tf.zeros(shape2, dtype = tf.float32), trainable = False, collections = [], name = 'state1')
                if not build_R_net:
                    src_reward = tf.Variable(tf.zeros(shape1, dtype = tf.float32), trainable = False, collections = [], name = 'reward')
        with network.root_scope(ROOT_SCOPE_MODEL(mod.id)) as root_scope:
            network.import_node(dev_root_scope, GLOABL_SCOPE_BATCH_PIPE + 'seq_len1',
                                GLOBAL_SCOPE_RUNTIME_PARAS + 'sequence_length')
            network.import_node(dev_root_scope, GLOABL_SCOPE_BATCH_PIPE + 'state1', s_name)
            if not build_R_net:
                network.import_node(dev_root_scope, GLOABL_SCOPE_BATCH_PIPE + 'reward', r_name)
            with network.scope(GLOBAL_SCOPE_Q_NET) as scope:
                config.Q_network_func(network = network, state_name = s_name, q_name = q_name, has_shadow = True)
                Q = network.get_tensor(scope + q_name)
                shadow_Q = network.get_tensor(scope + q_name, shadow = True)
            tf.identity(Q, name = q_name)
            tf.identity(shadow_Q, name = shadow_q_name)
            if build_R_net:
                with network.scope(GLOBAL_SCOPE_R_NET) as scope:
                    config.R_network_func(network = network, state_name = s_name, r_name = r_name)
                    R = network.get_tensor(scope + r_name)
                tf.identity(R, name = r_name)
            R = network.get_tensor(r_name)
            with network.scope(GLOABL_SCOPE_BATCH_PIPE) as scope:
                if config.double_dqn:
                    network.add_op_argmax(q_name, 'act_cand', axis = -1)
                    network.add_layer_select('select_shadow_q', shadow_q_name,
                                             'shadow_q_selected', scope + 'act_cand')
                else:
                    network.add_op_max(shadow_q_name, 'shadow_q_selected', axis = -1)
                sqe = network.add_layer_zero_seq_tail('zero_tail_shadow',
                                                      scope + 'shadow_q_selected', 'shadow_q_eval', low_dim = 0)
                q_target_out = tf.add(R, sqe * config.q_decay)
                q_target = tf.Variable(tf.zeros_like(q_target_out, dtype = tf.float32), trainable = False,
                                          collections = [], name = 'q_target')
                network.key_tensors['flows'].append(tf.assign(q_target, q_target_out, name = 'flow_q_target'))
                batch_ticket = tf.Variable(0, dtype = tf.int64, trainable = False, collections = [], name = 'batch_ticket')
                network.key_tensors['flows'].append(tf.assign(batch_ticket, src_batch_ticket, name = 'flow_batch_ticket'))
                seq_len = tf.Variable(tf.zeros(shape0, dtype = tf.int64), trainable = False, collections = [], name = 'seq_len')
                network.key_tensors['flows'].append(tf.assign(seq_len, src_seq_len0, name = 'flow_seq_len'))
                act_taken = tf.Variable(tf.zeros(shape1, dtype = tf.int64), trainable = False, collections = [], name = 'act_taken')
                network.key_tensors['flows'].append(tf.assign(act_taken, src_act_taken, name = 'flow_act_taken'))
                state = tf.Variable(tf.zeros(shape2, dtype = tf.float32), trainable = False, collections = [], name = 'state')
                network.key_tensors['flows'].append(tf.assign(state, src_state0, name = 'flow_state'))
            with network.scope(GLOBAL_SCOPE_TRAINED_STEP):
                network.key_tensors['trained_step'] = \
                    network.add_variable('number', tf.constant(0, dtype = tf.int64), trainable = False)
        network.session_new(target = serv.server.target, initialize = False, inter_op_threads = serv.tf_inter_op_threads)
    network.host = mod
    network.shadow_need_update = False
    network.q_target_expired = False
    return network


def do_flow(network:Network, timer:SectionTimer):
    mod:Model = network.host
    while not mod.initialized: sleep(IDLE_TIME_SHORT)
    flow_tensors = network.key_tensors['flows']
    fd = network._make_feed(feed_dict = None, dropout_rate_dict = None, learning_rate = 0,
                            queued_item_shape_dict = None, sequence_length = None, initial_state = None,
                            feed_initial_state = True, norm_in_batch = False)
    mod.locker.lock()
    timer.enter_timing_section()
    if network.shadow_need_update:
        network.session_update_shadow()
        network.shadow_need_update = False
        ret = FLAG_SHADOW_UPD
    else:
        ret = FLAG_NORMAL
    try: network._sess[0].run(flow_tensors, feed_dict = fd)
    except: pass
    network.q_target_expired = False
    timer.exit_timing_section()
    mod.locker.unlock()
    return ret