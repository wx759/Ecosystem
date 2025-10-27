__all__ = ['build', 'do_act']


from .__import__ import tf
from .__import__ import Network
from .__import__ import Model
from .__import__ import IDLE_TIME_SHORT
from .Scopes import *

from numpy import array
from time import sleep


def build(mod:Model) -> Network:
    network = Network()
    dev = mod.host
    serv = dev.host
    config = serv.config
    s_shape = config.s_shape
    q_name = 'reserved_node_Q'
    network.key_tensors = dict()
    with network.dispatch_to(dev.tf_code):
        with network.root_scope(ROOT_SCOPE_MODEL(mod.id)) as root_scope:
            state = tf.placeholder(dtype = tf.float32, shape = [1, 1] + s_shape, name = 'state')
            network.key_tensors['state'] = state
            tf.constant([1], dtype = tf.int64, name = GLOBAL_SCOPE_RUNTIME_PARAS + 'sequence_length')
            with network.scope(GLOBAL_SCOPE_Q_NET) as scope:
                config.Q_network_func(network = network, state_name = 'state',
                               q_name = q_name, has_shadow = False)
                Q = network.get_tensor(scope + q_name)
            config.act_num = tf.identity(Q, name = q_name).get_shape().as_list()[-1]
            network.add_op_argmax(q_name, 'arg_max_Q', axis = -1)
            network.key_tensors['act_cand'] = network.add_op_sum('arg_max_Q', 'act_cand')
            with network.scope(GLOBAL_SCOPE_TRAINED_STEP):
                network.key_tensors['trained_step'] = \
                    network.add_variable('number', tf.constant(0, dtype = tf.int64), trainable = False)
        network.session_new(target = serv.server.target, initialize = False, inter_op_threads = serv.tf_inter_op_threads)
    network.host = mod
    return network


def do_act(network:Network, state:array, initial_state:list):
    mod:Model = network.host
    while not mod.initialized: sleep(IDLE_TIME_SHORT)
    feed_dict = {network.key_tensors['state']: array([[state]])}

    tensors = [
        network.key_tensors['act_cand'],
        network.key_tensors['trained_step']
    ] + network.get_pull_essential_final_state()
    # print(feed_dict)
    # input()
    fd = network._make_feed(feed_dict = feed_dict, dropout_rate_dict = None,
                            learning_rate = 0, queued_item_shape_dict = None, sequence_length = None,
                            initial_state = initial_state, feed_initial_state = True, norm_in_batch = False)
    mod.locker.lock()
    try: ret = network._sess[0].run(tensors, feed_dict = fd)
    except: pass
    mod.locker.unlock()
    if ret is None: return None
    act_cand = ret[0]
    trained_step = ret[1]
    final_state = ret[2 : ]
    if len(final_state) == 0: final_state = None
    return (act_cand, final_state, trained_step)