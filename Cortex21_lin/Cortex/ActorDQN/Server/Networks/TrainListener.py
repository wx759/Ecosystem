__all__ = ['build', 'do_train']


from .__import__ import tf
from .__import__ import Network
from .__import__ import Model
from .__import__ import get_section_timer
from .__import__ import SectionTimer
from .__import__ import FLAG_NO_BATCH
from .__import__ import wait_notifier
from .__import__ import notify
from .Scopes import *


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
    q_name = 'reserved_node_Q'
    network.key_tensors = dict()
    with network.dispatch_to(dev.tf_code):
        with network.root_scope(ROOT_SCOPE_MODEL(mod.id)) as root_scope:
            with network.scope(GLOABL_SCOPE_BATCH_PIPE):
                network.key_tensors['batch_ticket'] = \
                    tf.Variable(0, dtype = tf.int64, trainable = False, collections = [], name = 'batch_ticket')
                seq_len = tf.Variable(tf.zeros(shape0, dtype = tf.int64), trainable = False, collections = [], name = 'seq_len')
                tf.Variable(tf.zeros(shape1, dtype = tf.int64), trainable = False, collections = [], name = 'act_taken')
                tf.Variable(tf.zeros(shape2, dtype = tf.float32), trainable = False, collections = [], name = 'state')
                tf.Variable(tf.zeros(shape1, dtype = tf.float32), trainable = False, collections = [], name = 'q_target')
            tf.identity(seq_len, GLOBAL_SCOPE_RUNTIME_PARAS + 'sequence_length')
            with network.scope(GLOBAL_SCOPE_Q_NET) as scope:
                config.Q_network_func(network = network, state_name = GLOABL_SCOPE_BATCH_PIPE + 'state',
                               q_name = q_name, has_shadow = False)
                Q = network.get_tensor(scope + q_name)
            tf.identity(Q, name = q_name)
            with network.scope(GLOBAL_SCOPE_Q_LEARNER) as scope:
                network.add_layer_select('select_q', q_name, 'q_selected', GLOABL_SCOPE_BATCH_PIPE + 'act_taken')
                network.add_layer_zero_seq_tail('zero_tail_eval', scope + 'q_selected', 'q_eval', low_dim = 0)
                network.add_layer_zero_seq_tail('zero_tail_target', GLOABL_SCOPE_BATCH_PIPE + 'q_target', 'q_target', low_dim = 0)
                network.key_tensors['td_error'] = \
                    network.add_reduce_mean_square_error('td_error', scope + 'q_target', scope + 'q_eval', axis = 1)
                network.add_op_mean(scope + 'td_error', 'mse')
                network.set_trainer(config.optimizer, scope + 'mse')
            with network.scope(GLOBAL_SCOPE_TRAINED_STEP):
                step_num = network.add_variable('number', tf.constant(0, dtype = tf.int64), trainable = False)
                network.key_tensors['inc_step_num'] = tf.assign(step_num, step_num + 1, name = 'inc_step_num')
        network.session_new(target = serv.server.target, initialize = False, inter_op_threads = serv.tf_inter_op_threads)
    network.host = mod
    return network


def do_train(network:Network, learning_rate:float, dropout_rate_dict:dict):
    mod:Model = network.host
    batch_pipe_notifier = mod.train_listener.notifiers['batch_pipe']
    info = wait_notifier(batch_pipe_notifier)
    if info:
        flag, t_timer = info
        timer:SectionTimer = get_section_timer(t_timer)
        tensors = [
                      network.key_tensors['batch_ticket'],
                      network.key_tensors['td_error'],
                      network.key_tensors['inc_step_num']
                  ] + network.get_pull_essential_trainer()
        fd = network._make_feed(feed_dict = None, dropout_rate_dict = dropout_rate_dict,
                                learning_rate = learning_rate, queued_item_shape_dict = None,
                                sequence_length = None, initial_state = None, feed_initial_state = True,
                                norm_in_batch = True)
        batch_pipe_net = mod.batch_pipe.network
        mod.locker.lock()
        timer.enter_timing_section()
        if batch_pipe_net.q_target_expired:
            batch_pipe_net.q_target_expired = False
            ret = FLAG_NO_BATCH
        else:
            try: ret = network._sess[0].run(tensors, feed_dict = fd)
            except: pass
            if ret is not None:
                trained_step = ret[2]
                config = mod.host.host.config
                if trained_step % config.upd_shadow_period == 0:
                    batch_pipe_net.shadow_need_update = True
                ret = (ret[0], ret[1], flag, trained_step)
        timer.exit_timing_section()
        notify(batch_pipe_notifier, t_timer)
        mod.locker.unlock()
    else:
        ret = None
    return ret