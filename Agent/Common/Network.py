__all__ = ['Optimizer', 'ActivationFunc', 'VarFilter', 'TF_Neural_Network', 'ScalarLogger',
           'GLOBAL_SCOPE_SHADOW', 'GLOBAL_SCOPE_SHADOW_ASSIGNERS', 'GLOBAL_SCOPE_INITIAL_STATE',
           'GLOBAL_SCOPE_VARIABLE_LOADERS', 'GLOBAL_SCOPE_RUNTIME_PARAS','ActivationFunc']


import numpy

from os import environ
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .Funcs import is_list
from .Funcs import get_file_path_from_name
from .Funcs import make_tensorbord_runner as func_mtr
from .Locker import BasicLock

import tensorflow as tf_root
__tfv__ = tf_root.__version__
if __tfv__ >= '1.14':
    tf = tf_root.compat.v1
    tf.where = tf_root.compat.v2.where
else:
    tf = tf_root

from tensorflow.python.framework.errors_impl import CancelledError
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from tensorflow.python.client import device_lib
from functools import reduce
from numpy import array
from copy import deepcopy as COPY
from threading import Thread
from shutil import rmtree
from os import makedirs
from _pickle import load as load_pack
from _pickle import dump as dump_pack


_DEFAULT_LOGS_PATH = 'logs'
_DEFAULT_MOVING_AVG_DECAY = 0.95

GLOBAL_SCOPE_SHADOW = 'shadow/'
GLOBAL_SCOPE_SHADOW_ASSIGNERS = GLOBAL_SCOPE_SHADOW + 'assigners/'
GLOBAL_SCOPE_INITIAL_STATE = 'initial_state/'
GLOBAL_SCOPE_VARIABLE_LOADERS = 'variable_loaders/'
GLOBAL_SCOPE_RUNTIME_PARAS = 'runtime_paras/'

_LOCAL_SCOPE_SHAPE_IN = 'shape_in'
_LOCAL_SCOPE_SHAPE_OUT = 'shape_out'
_LOCAL_SCOPE_BATCH_NORM = 'batch_norm'
_LOCAL_SCOPE_LSTM_THROUGH_SEQ = 'lstm_through_seq'
_LOCAL_SCOPE_FINAL_STATE = 'final_state'

_INTERNAL_QUEUE_CAPACITY = 64

def _tensor_shape_model(X):
    try: shape = X.get_shape().as_list()
    except ValueError: shape = list()
    return [-1 if s is None else s for s in shape]

def _tensor_shape_runtime(X): return tf.shape(X, out_type = tf.int64)

def _is_reshapable(shape):
    if shape is None: return False
    if not isinstance(shape, list) \
            and not isinstance(shape, tuple)\
            and not isinstance(shape, array): return False
    if len(shape) == 0: return True
    if not isinstance(shape[0], int) or shape[0] == 0 or shape[0] < -1: return False
    if len(shape) == 1: return True
    return reduce(bool.__and__, [(isinstance(s, int) and s > 0) for s in shape[1 : ]])

def _reorder_rnn_state(src_state, src_in_batch_order:bool = False, state_template:list = None):
    if src_in_batch_order:
        if src_state is None: return None
        batch_size = len(src_state)
        if state_template is None: state_template = []
        tensor_num = len(state_template)
        dst = []
        for tensor in state_template:
            zero_state = [0.0] * _tensor_shape_model(tensor)[-1]
            dst.append(array([zero_state] * batch_size, dtype = numpy.float32))
        for i in range(batch_size):
            row = src_state[i]
            if row is not None:
                row = COPY(row)
                for t, v in zip(range(tensor_num), row): dst[t][i] = v
        return tuple(dst)
    else:
        if src_state is None: return None
        else: return tuple(map(tuple, zip(*src_state)))


class Optimizer:
    Adadelta = tf.train.AdadeltaOptimizer
    AdagradDA = tf.train.AdagradDAOptimizer
    Adagrad = tf.train.AdagradOptimizer
    Adam = tf.train.AdamOptimizer
    Ftrl = tf.train.FtrlOptimizer
    GradientDescent = tf.train.GradientDescentOptimizer
    Momentum = tf.train.MomentumOptimizer
    ProximalAdagrad = tf.train.ProximalAdagradOptimizer
    ProximalGradientDescent = tf.train.ProximalGradientDescentOptimizer
    RMSProp = tf.train.RMSPropOptimizer
    SyncReplicas = tf.train.SyncReplicasOptimizer


class ActivationFunc:
    crelu = tf.nn.crelu
    elu = tf.nn.elu
    leaky_relu = tf.nn.leaky_relu
    relu = tf.nn.relu
    relu6 = tf.nn.relu6
    selu = tf.nn.selu
    sigmoid = tf.nn.sigmoid
    softmax = tf.nn.softmax
    softplus = tf.nn.softplus
    softsign = tf.nn.softsign
    tanh = tf.nn.tanh


class VarFilter:
    @staticmethod
    def for_trainer_only(name:str, var_info:dict):
        return var_info['for_trainer']

    @staticmethod
    def not_for_trainer(name:str, var_info:dict):
        return not VarFilter.for_trainer_only(name, var_info)

    @staticmethod
    def shadow_only(name:str, var_info:dict = None):
        return name.find(GLOBAL_SCOPE_SHADOW) >= 0

    @staticmethod
    def not_shadow(name:str, var_info:dict = None):
        return not VarFilter.shadow_only(name, var_info)

    @staticmethod
    def filter_value_dict(value_dict:dict, var_filters):
        if hasattr(var_filters, '__call__'): var_filters = [var_filters]
        return {name: value for name, value in value_dict.items() if
                (var_filters is None or reduce(bool.__and__, [var_filter(name, None) for var_filter in var_filters]))}


class TF_Neural_Network:
    class ScopeContext():
        def __init__(self, network, name:str = None):
            self.network = network
            if name == '' or name == '/': name = None
            self.name = name

        def __enter__(self):
            network = self.network
            self._context_graph_as_default = network.graph.as_default()
            self._context_graph_as_default.__enter__()
            self._context_name_scope = None
            self._context_non_variable_scope = None
            self._context_variable_scope = None
            if self.name is None or self.name[-1] == '/':
                self._context_non_variable_scope = tf.variable_scope(network._none_scope['variable_scope'],
                                                                     reuse = tf.AUTO_REUSE, auxiliary_name_scope = False)
                self._context_non_variable_scope.__enter__()
                new_name_scope_str = network._current_root['name_scope']
                new_variable_scope_str = network._current_root['variable_scope']
                if self.name is not None:
                    new_name_scope_str += self.name
                    new_variable_scope_str += self.name
                if new_name_scope_str == '': new_name_scope = None
                else: new_name_scope = new_name_scope_str
                if new_variable_scope_str == '': new_variable_scope = None
                else: new_variable_scope = new_variable_scope_str[ : -1]
                self._context_name_scope = tf.name_scope(new_name_scope)
                self._context_name_scope.__enter__()
                if new_variable_scope is not None:
                    self._context_variable_scope = tf.variable_scope(new_variable_scope,
                                                                     reuse = tf.AUTO_REUSE, auxiliary_name_scope = False)
                    self._context_variable_scope.__enter__()
            else:
                self._context_variable_scope = tf.variable_scope(self.name, reuse = tf.AUTO_REUSE, auxiliary_name_scope = True)
                self._context_variable_scope.__enter__()
                new_name_scope_str = tf.get_default_graph().get_name_scope() + '/'
            return new_name_scope_str[len(network._current_root['name_scope']) : ]

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._context_variable_scope: self._context_variable_scope.__exit__(exc_type, exc_val, exc_tb)
            if self._context_name_scope: self._context_name_scope.__exit__(exc_type, exc_val, exc_tb)
            if self._context_non_variable_scope: self._context_non_variable_scope.__exit__(exc_type, exc_val, exc_tb)
            self._context_graph_as_default.__exit__(exc_type, exc_val, exc_tb)


    class RootScopeContext():
        def __init__(self, network, name:str = None):
            self.network = network
            if name == '' or name == '/': name = None
            self.name = name

        def __enter__(self):
            network = self.network
            self.old_root = network._current_root
            self._context_graph_as_default = network.graph.as_default()
            self._context_graph_as_default.__enter__()
            self._context_non_name_scope = tf.name_scope(network._none_scope['name_scope'])
            self._context_non_name_scope.__enter__()
            self._context_non_variable_scope = tf.variable_scope(network._none_scope['variable_scope'],
                                                                 reuse = tf.AUTO_REUSE, auxiliary_name_scope = False)
            self._context_non_variable_scope.__enter__()
            if self.name is not None:
                name_without_slash = self.name
                if name_without_slash[-1] == '/': name_without_slash = name_without_slash[: -1]
                self._context_name_scope = tf.name_scope(self.name)
                new_name_scope = self._context_name_scope.__enter__()
                self._context_variable_scope = tf.variable_scope(name_without_slash,
                                                             auxiliary_name_scope = False)
                self._context_variable_scope.__enter__()
                new_variable_scope = name_without_slash + '/'
            else:
                self._context_name_scope = self._context_variable_scope = None
                new_name_scope = new_variable_scope = ''
            network._current_root = {'name_scope': new_name_scope, 'variable_scope': new_variable_scope}
            return new_name_scope

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.network._current_root = self.old_root
            if self._context_variable_scope: self._context_variable_scope.__exit__(exc_type, exc_val, exc_tb)
            if self._context_name_scope: self._context_name_scope.__exit__(exc_type, exc_val, exc_tb)
            self._context_non_variable_scope.__exit__(exc_type, exc_val, exc_tb)
            self._context_non_name_scope.__exit__(exc_type, exc_val, exc_tb)
            self._context_graph_as_default.__exit__(exc_type, exc_val, exc_tb)

    class CopyVarContext():
        def __init__(self, network, copy_from:str = None):
            self.network = network
            if copy_from == '' or copy_from == '/': copy_from = None
            if copy_from != '' and copy_from[-1] != '/': copy_from += '/'
            self.copy_from = copy_from

        def __enter__(self):
            network = self.network
            self.old_copy_from = network._current_copy_from
            network._current_copy_from = self.copy_from

        def __exit__(self, exc_type, exc_val, exc_tb):
            network = self.network
            network._current_copy_from = self.old_copy_from


    def __init__(self):
        self.graph = tf.Graph()
        self._current_root = {
            'name_scope': '',
            'variable_scope': ''
        }
        self._current_copy_from = ''
        with self.graph.as_default():
            self._none_scope = {
                'name_scope': None,
                'variable_scope': tf.get_variable_scope()
            }
        self._global_runtime_parameters = dict()
        self._var_dict = dict()
        self._shadow_assigners = {
            'names': list(),
            'tensors': list()
        }
        self._batch_norm_paras_updaters = {
            'names': list(),
            'tensors': list()
        }
        self._queues = {}
        self._queued_initial_state = []
        self._initial_state_size = []
        self._rnn_state_queues = []
        self._final_state = []
        self._final_state_shadow = []
        self._dropout_rates = []
        self._sess = {}
        self._sess_mutex = {}
        self._pub_mutex = BasicLock()
        self._sess_head = -1
        self._trainer = None
        self._initializer_global = None
        self._initializer_trainer = None
        self._logs_path = _DEFAULT_LOGS_PATH
        self._merge_summary = None

    def _get_session(self, h_sess = None):
        if h_sess is None:
            all_sess = list(self._sess.values())
            if len(all_sess) == 0: return None
            if len(all_sess) == 1: return all_sess[0]
            return all_sess
        if is_list(h_sess): return [_get_session(st) for st in h_sess]
        try: return self._sess[h_sess]
        except: return None

    def _make_feed(self, feed_dict:dict, dropout_rate_dict:dict, learning_rate:float, queued_item_shape_dict:dict,
                   sequence_length:array, initial_state:tuple, feed_initial_state:bool, norm_in_batch:bool):
        if feed_dict is None: feed_dict = {}
        if dropout_rate_dict is None: dropout_rate_dict = {}
        if queued_item_shape_dict is None: queued_item_shape_dict = {}
        global_runtime_parameters = self._global_runtime_parameters
        fd = dict()
        for k, v in global_runtime_parameters.items():
            default_feed = v['default_feed']
            tensor = v['tensor']
            if default_feed is not None:
                fd[tensor] = default_feed
        for k, v in dropout_rate_dict.items():
            try: tensor = global_runtime_parameters[k]['tensor']
            except: tensor = None
            if tensor is not None: fd[tensor] = v
        if learning_rate > 0 and 'learning_rate' in global_runtime_parameters.keys():
            tensor = global_runtime_parameters['learning_rate']['tensor']
            fd[tensor] = learning_rate
        if 'norm_in_batch' in global_runtime_parameters.keys():
            tensor = global_runtime_parameters['norm_in_batch']['tensor']
            fd[tensor] = norm_in_batch
        if sequence_length is not None and 'sequence_length' in global_runtime_parameters.keys():
            tensor = global_runtime_parameters['sequence_length']['tensor']
            fd[tensor] = array(sequence_length, dtype = numpy.int64)
        if 'zero_initial_state' in global_runtime_parameters.keys():
            zero_flag_tensor = global_runtime_parameters['zero_initial_state']['tensor']
            fd[zero_flag_tensor] = False
            if feed_initial_state:
                if initial_state is None:
                    fd[zero_flag_tensor] = True
                    for tensor in self._queued_initial_state:
                        fd[tensor] = array([0], dtype = numpy.float32)
                else:
                    for tensor, size, src in zip(self._queued_initial_state, self._initial_state_size, initial_state):
                        zero_state = [0.0] * size
                        row = []
                        for v in src:
                            row = numpy.concatenate((row, (zero_state if v is None else v)))
                        fd[tensor] = array(row, dtype = numpy.float32)
        for k, v in queued_item_shape_dict.items():
            out_shape_tensor = self._queues[k]['out_shape']
            if out_shape_tensor is not None: fd[out_shape_tensor] = v
            try:
                shadow_name = self._queues[k]['root_scope'] + GLOBAL_SCOPE_SHADOW + k[self._queues[k]['root_scope'] : ]
                out_shape_tensor = self._queues[shadow_name]['out_shape']
                if out_shape_tensor is not None: fd[out_shape_tensor] = v
            except KeyError: pass
        for k, v in feed_dict.items():
            fd[((k + ':0') if isinstance(k, str) else k)] = array(v)
        return fd

    def _current_name_scope(self):
        with self.graph.as_default():
            current_name_scope = self.graph.get_name_scope()
            if current_name_scope != '': current_name_scope += '/'
            return current_name_scope[len(self._current_root['name_scope']) : ]

    def _current_variable_scope(self):
        with self.graph.as_default():
            current_variable_scope = tf.get_variable_scope().name
            if current_variable_scope != '': current_variable_scope += '/'
            return current_variable_scope[len(self._current_root['variable_scope']) : ]

    def _get_tensor_from_none_root(self, node):
        with self.graph.as_default():
            multi_nodes = is_list(node)
            if not multi_nodes: node = [node]
            t_num = len(node)
            tensor = [None] * t_num
            for i in range(t_num):
                if isinstance(node[i], str):
                    tensor[i] = None
                    try: tensor[i] = self.graph.get_tensor_by_name(node[i] + ':0')
                    except: pass
                    if tensor[i] is None:
                        try: tensor[i] = self.graph.get_tensor_by_name(node[i])
                        except: pass
                    if tensor[i] is None:
                        try: tensor[i] = self.graph.get_operation_by_name(node[i])
                        except: pass
                else: tensor[i] = node[i]
            return tensor if multi_nodes else tensor[0]

    def _get_sequence_length(self, shape_high:tf.Tensor):
        tensor = self.get_tensor(GLOBAL_SCOPE_RUNTIME_PARAS + 'sequence_length')
        if tensor is None:
            try: tensor = self._global_runtime_parameters['sequence_length']['tensor']
            except KeyError: tensor = None
        if tensor is None:
            seq_len_shape = [None] * (_tensor_shape_model(shape_high)[0] - 1)
            with self.scope(GLOBAL_SCOPE_RUNTIME_PARAS):
                tensor = tf.placeholder(tf.int64, shape = seq_len_shape, name = 'sequence_length')
                self._global_runtime_parameters['sequence_length'] = {'tensor': tensor, 'default_feed': None}
        return tensor

    def _get_runtime_parameter(self, name:str, dtype, default_feed):
        tensor = self.get_tensor(GLOBAL_SCOPE_RUNTIME_PARAS + name)
        if tensor is None:
            try: tensor = self._global_runtime_parameters[name]['tensor']
            except KeyError: tensor = None
        if tensor is None:
            with self.scope(GLOBAL_SCOPE_RUNTIME_PARAS):
                tensor = tf.placeholder(dtype = dtype, name = name)
                self._global_runtime_parameters[name] = {'tensor': tensor, 'default_feed': default_feed}
        return tensor

    def _get_initial_state_tensors(self, initial_state_scope, final_state_scope, X, y_size):
        initial_c = self.get_tensor(initial_state_scope + 'c')
        initial_h = self.get_tensor(initial_state_scope + 'h')
        if initial_c is None:
            zero_initial_state = self._get_runtime_parameter('zero_initial_state', tf.bool, None)
            final_c_name = final_state_scope + 'c'
            final_h_name = final_state_scope + 'h'
            root_name_scope = self._current_root['name_scope']
            with self.scope(initial_state_scope):
                x_shape = _tensor_shape_runtime(X)
                state_shape = tf.cast((x_shape[0], y_size), dtype = tf.int64)
                zero_state = tf.zeros(shape = state_shape, dtype = tf.float32, name = 'zero_state')
                c_queued = self.add_queue('c_queue', _INTERNAL_QUEUE_CAPACITY, final_c_name, 'c_queued',
                                                          out_shape = [-1], dtype = tf.float32)
                self._rnn_state_queues.append(root_name_scope + initial_state_scope + 'c_queue')
                self._queued_initial_state.append(c_queued)
                self._initial_state_size.append(y_size)
                initial_c = tf.reshape(tf.cond(zero_initial_state, lambda: zero_state, lambda: c_queued), state_shape, name = 'c')
                h_queued = self.add_queue('h_queue', _INTERNAL_QUEUE_CAPACITY, final_h_name, 'h_queued',
                                                          out_shape = [-1], dtype = tf.float32)
                self._rnn_state_queues.append(root_name_scope + initial_state_scope + 'h_queue')
                self._queued_initial_state.append(h_queued)
                self._initial_state_size.append(y_size)
                initial_h = tf.reshape(tf.cond(zero_initial_state, lambda: zero_state, lambda: h_queued), state_shape, name = 'h')
        return initial_c, initial_h

    def _make_push_op(self, queue_name, input):
        Q = self._queues[queue_name]['queue_obj']
        with self.root_scope(self._queues[queue_name]['root_scope']):
            with tf.name_scope(self._queues[queue_name]['base_scope']):
                if not isinstance(input, tf.Tensor):
                    in_tensor = self.get_tensor(input)
                    if in_tensor is None: return input
                    with tf.name_scope(queue_name + '/'): input = tf.identity(in_tensor)
            with tf.name_scope(queue_name + '/'):
                return Q.enqueue(input, name = 'push')

    def _get_base_variable(self, var:tf.Variable):
        base_var = None
        if self._current_copy_from != '' and self._current_copy_from != self._current_root['variable_scope']:
            var_name_from_root = var.name[len(self._current_root['variable_scope']) : ]
            base_var_name = self._current_copy_from + var_name_from_root
            try:
                base_var = self._var_dict[base_var_name]['var']
            except: raise ValueError('Base variable in copy-root not found.')
        if base_var == var: base_var = None
        return base_var

    def _append_var_copy(self, base_var:tf.Variable, var:tf.Variable):
        base_rec = self._var_dict[base_var.name]
        if var.name in base_rec['copy_names']: return
        base_rec['copy_names'].append(var.name)
        base_rec['loaders'].append(tf.assign(var, base_rec['src']))
        base_rec['copy'].append(tf.assign(var, base_var))

    def _complete_variable(self, var:tf.Variable, trainable:bool, has_shadow:bool, for_trainer:bool):
        if var is None: return
        base_var = self._get_base_variable(var)
        with self.scope(GLOBAL_SCOPE_VARIABLE_LOADERS):
            if base_var is None:
                if var.name not in self._var_dict.keys():
                    src = tf.placeholder(var.dtype, _tensor_shape_model(var))
                    loader = tf.assign(var, src)
                    self._var_dict[var.name] = {
                        'var':var,
                        'src':src,
                        'loaders': [loader],
                        'for_trainer': for_trainer,
                        'copy_names': list(),
                        'copy': list()
                    }
            else:
                self._append_var_copy(base_var, var)
        if has_shadow: self._make_shadow_assigner(var)
        if not trainable:
            trainable_collection = self.graph.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
            try: trainable_collection.remove(var)
            except ValueError: pass

    def _make_shadow_assigner(self, var:tf.Variable):
        if var.name in self._shadow_assigners['names']: return
        root_variable_scope = self._current_root['variable_scope']
        var_name = var.name[len(root_variable_scope) : ]
        shadow_var_name = GLOBAL_SCOPE_SHADOW + var_name
        with self.scope(GLOBAL_SCOPE_SHADOW_ASSIGNERS):
            shadow_var = self.get_tensor(shadow_var_name)
            self._shadow_assigners['names'].append(var.name)
            self._shadow_assigners['tensors'].append(tf.assign(shadow_var, var))

    def _get_variable_by_name(self, var_name:str):
        with self.graph.as_default():
            vars = tf.global_variables()
            for v in vars:
                if v.name == var_name: return v
            return None

    def _shape_in(self, X:tf.Tensor, low_dim:int, as_seq:bool = False):
        with self.scope(_LOCAL_SCOPE_SHAPE_IN):
            x = tf.identity(X)
            shape_x = _tensor_shape_model(x)
            dim_num = len(shape_x)
            shape_high = _tensor_shape_runtime(x)[0 : dim_num - low_dim]
            shape_low = shape_x[dim_num - low_dim : ]
            new_shape_uncat = [[-1], shape_high[-1 : ]] if as_seq else [[-1]]
            if len(shape_low) > 0: new_shape_uncat.append(shape_low)
            new_shape = tf.concat(new_shape_uncat, 0)
            return tf.reshape(x, new_shape), shape_high

    def _shape_out(self, X:tf.Tensor, shape_high:tf.Tensor, as_seq:bool = False):
        with self.scope(_LOCAL_SCOPE_SHAPE_OUT):
            x = tf.identity(X)
            shape_low = _tensor_shape_model(x)[2 : ] if as_seq else _tensor_shape_model(x)[1 : ]
            new_shape_uncat = [shape_high]
            if len(shape_low) > 0: new_shape_uncat.append(shape_low)
            new_shape = tf.concat(new_shape_uncat, 0)
            return tf.reshape(x, new_shape)

    def _add_shadow_unit(self, unit_func, para_list:tuple, inputs_at:tuple):
        para_list = list(para_list)
        for i in inputs_at:
            x_name = para_list[i]
            x_name = self.get_tensor(x_name).name
            if x_name[-2 : ] == ':0': x_name = x_name[ : -2]
            if x_name.find(self._current_root['name_scope']) == 0:
                x_name = x_name[len(self._current_root['name_scope']) : ]
            else:
                if x_name.find(self._current_root['variable_scope']) == 0:
                    x_name = x_name[len(self._current_root['variable_scope']) : ]
            shadow_x_name = GLOBAL_SCOPE_SHADOW + x_name
            if self.get_tensor(shadow_x_name) is None: shadow_x_name = x_name
            para_list[i] = shadow_x_name
        shadow_name_scope = self._current_root['name_scope'] + GLOBAL_SCOPE_SHADOW + self._current_name_scope()
        shadow_variable_scope = self._current_root['variable_scope'] + GLOBAL_SCOPE_SHADOW + self._current_variable_scope()
        if shadow_variable_scope != '': shadow_variable_scope = shadow_variable_scope[ : -1]
        with self.graph.as_default():
            with tf.name_scope(shadow_name_scope) as scope:
                with tf.variable_scope(self._none_scope['variable_scope'], auxiliary_name_scope = False):
                    with tf.variable_scope(shadow_variable_scope, reuse = True, auxiliary_name_scope = False):
                        return unit_func(*para_list)

    def _thread_func_solve_sess(self, solve_func, idx:int, h_sess:list, para_list:tuple,
                                h_sess_at:int, extra_list_para_at:tuple, ret:list):
        para_num = len(para_list)
        z = zip(range(para_num), para_list)
        sub_para_list = [(p[idx] if (is_list(p) and (j in extra_list_para_at)) else p) for j, p in z]
        sub_para_list[h_sess_at] = h_sess[idx]
        ret[idx] = solve_func(*sub_para_list)

    def _solve_none_and_multi_sess(self, solve_func, para_list:tuple, h_sess_at:int, extra_list_para_at:tuple = None,
                                   reduce_func = None, multi_thread:bool = False):
        if extra_list_para_at is None: extra_list_para_at = ()
        h_sess = para_list[h_sess_at]
        if h_sess is None:
            h_sess = list(self._sess.keys())
            if len(h_sess) == 0: return None
            if len(h_sess) == 1: return h_sess[0]
        if is_list(h_sess):
            sess_num = len(h_sess)
            ret = [None] * sess_num
            if multi_thread and sess_num > 1:
                threads = [Thread(daemon = True, target = self._thread_func_solve_sess,
                                            args = (solve_func, idx, h_sess, para_list, h_sess_at, extra_list_para_at, ret))
                                for idx in range(sess_num)]
                for t in threads: t.start()
                for t in threads: t.join()
            else:
                for idx in range(sess_num):
                    self._thread_func_solve_sess(solve_func, idx, h_sess, para_list, h_sess_at, extra_list_para_at, ret)
            return (ret if reduce_func is None else reduce(reduce_func, ret))
        else: return None

    def _batch_norm(self, X:tf.Tensor, moving_avg_decay:float = _DEFAULT_MOVING_AVG_DECAY,
                    trainable:bool = True, has_shadow:bool = False):
        norm_in_batch = self._get_runtime_parameter('norm_in_batch', tf.bool, None)
        x_shape = _tensor_shape_model(X)
        axis = list(range(len(x_shape) - 1))
        batch_mean, batch_var = tf.nn.moments(X, axis)
        para_shape = x_shape[-1 : ]
        global_mean = self.add_variable('global_mean', tf.zeros(para_shape, dtype = tf.float32),
                                        trainable = False, has_shadow = has_shadow)
        global_var = self.add_variable('global_var', tf.ones(para_shape, dtype = tf.float32),
                                       trainable = False, has_shadow = has_shadow)
        in_shadow = (self._current_variable_scope().find(GLOBAL_SCOPE_SHADOW) == 0)
        if not in_shadow and global_mean.name not in self._batch_norm_paras_updaters['names']:
            update_global_mean = tf.assign(global_mean,
                                           global_mean * moving_avg_decay + batch_mean * (1 - moving_avg_decay),
                                           name = 'update_global_mean')
            update_global_var = tf.assign(global_var,
                                          global_var * moving_avg_decay + batch_var * (1 - moving_avg_decay),
                                          name = 'update_global_var')
            self._batch_norm_paras_updaters['names'].append(global_mean.name)
            self._batch_norm_paras_updaters['tensors'].append(update_global_mean)
            self._batch_norm_paras_updaters['tensors'].append(update_global_var)
        offset = self.add_variable('offset', tf.zeros(para_shape), trainable = trainable, has_shadow = has_shadow)
        scale = self.add_variable('scale', tf.ones(para_shape), trainable = trainable, has_shadow = has_shadow)
        provNaN = 1e-10
        mean, var = tf.cond(norm_in_batch, lambda: (batch_mean, batch_var), lambda: (global_mean, global_var))
        return tf.nn.batch_normalization(X, mean, var, offset, scale, provNaN)

    @staticmethod
    def get_gpu_info():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        devices = device_lib.list_local_devices(session_config = config)
        return [d for d in devices if d.device_type == 'GPU']

    def dispatch_to(self, device_name:str):
        return self.graph.device(device_name)

    def is_current_default(self):
        return (tf.get_default_graph() == self.graph)

    def root_scope(self, root_name:str = None):
        return TF_Neural_Network.RootScopeContext(self, root_name)

    def scope(self, scope_name:str):
        return TF_Neural_Network.ScopeContext(self, scope_name)

    def copy_variables_from(self, copy_from:str):
        return TF_Neural_Network.CopyVarContext(self, copy_from)

    def import_node(self, from_root:str, node:str, name:str):
        with self.graph.as_default():
            tensor = self.graph.get_tensor_by_name(from_root + node + ':0')
            return tf.identity(tensor, name)

    def add_variable(self, name:str, init_value:tf.Tensor, trainable:bool = True, has_shadow:bool = False):
        with self.graph.as_default():
            var = tf.get_variable(name = name, initializer = init_value, dtype = init_value.dtype, trainable = trainable)
            self._complete_variable(var, trainable, has_shadow, False)
            return var

    def add_placeholder(self, name:str, shape:list = None, dtype = tf.float32):
        with self.graph.as_default():
            return tf.placeholder(dtype, shape = shape, name = name)

    def add_op_reshape(self, x_name:str, y_name:str, shape, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow: self._add_shadow_unit(self.add_op_reshape, (x_name, y_name, shape), (0,))
            X = self.get_tensor(x_name)
            Y = tf.reshape(X, shape, name = y_name)
            return Y

    def add_op_argmax(self, x_name:str, y_name:str, axis = None, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow: self._add_shadow_unit(self.add_op_argmax, (x_name, y_name, axis), (0,))
            X = self.get_tensor(x_name)
            Y = tf.argmax(X, axis, name = y_name)
            return Y

    def add_op_argmin(self, x_name:str, y_name:str, axis = None, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow: self._add_shadow_unit(self.add_op_argmin, (x_name, y_name, axis), (0,))
            X = self.get_tensor(x_name)
            Y = tf.argmin(X, axis, name = y_name)
            return Y

    def add_op_max(self, x_name:str, y_name:str, axis = None, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow: self._add_shadow_unit(self.add_op_max, (x_name, y_name, axis), (0,))
            X = self.get_tensor(x_name)
            Y = tf.reduce_max(X, axis, name = y_name)
            return Y

    def add_op_min(self, x_name:str, y_name:str, axis = None, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow: self._add_shadow_unit(self.add_op_min, (x_name, y_name, axis), (0,))
            X = self.get_tensor(x_name)
            Y = tf.reduce_min(X, axis, name = y_name)
            return Y

    def add_op_mean(self, x_name:str, y_name:str, axis = None, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow: self._add_shadow_unit(self.add_op_mean, (x_name, y_name, axis), (0,))
            X = self.get_tensor(x_name)
            Y = tf.reduce_mean(X, axis, name = y_name)
            return Y

    def add_op_sum(self, x_name:str, y_name:str, axis = None, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow: self._add_shadow_unit(self.add_op_sum, (x_name, y_name, axis), (0,))
            X = self.get_tensor(x_name)
            Y = tf.reduce_sum(X, axis, name = y_name)
            return Y

    def add_op_transpose(self, x_name:str, y_name:str, dim_order:list, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow: self._add_shadow_unit(self.add_op_transpose, (x_name, y_name, dim_order), (0,))
            X = self.get_tensor(x_name)
            Y = tf.transpose(X, dim_order, name = y_name)
            return Y

    def add_reduce_average_cross_entropy(self, scalar_name:str, lab_name:str, out_name:str, log_scalar:bool = False):
        with self.graph.as_default():
            lab = self.get_tensor(lab_name)
            out = self.get_tensor(out_name)
            provNaN = 1e-10
            with self.scope('reduce_ACE'):
                ace = -tf.reduce_mean((lab + provNaN) * tf.log(out + provNaN))
            ace = tf.identity(ace, name = scalar_name)
            if log_scalar: tf.summary.scalar('scalar.' + scalar_name, ace)
            return ace

    def add_reduce_mean_square_error(self, scalar_name:str, expected_name:str, out_name:str, axis = None, log_scalar:bool = False):
        with self.graph.as_default():
            expected = self.get_tensor(expected_name)
            out = self.get_tensor(out_name)
            with self.scope('reduce_MSE'):
                mse = tf.reduce_mean(tf.square(out - expected), axis = axis)
            mse = tf.identity(mse, name = scalar_name)
            if len(_tensor_shape_model(mse)) > 0: log_scalar = False
            if log_scalar: tf.summary.scalar('scalar.' + scalar_name, mse)
            return mse

    def add_reduce_classifier_accuracy(self, scalar_name:str, lab_name:str, out_name:str, log_scalar:bool = False):
        with self.graph.as_default():
            lab = self.get_tensor(lab_name)
            out = self.get_tensor(out_name)
            with self.scope('reduce_accuracy'):
                lab_num = tf.argmax(lab, -1)
                out_num = tf.argmax(out, -1)
                hit = tf.cast(tf.equal(lab_num, out_num), tf.int64)
                acc = tf.reduce_sum(hit)/tf.reduce_prod(_tensor_shape_runtime(hit))
            acc = tf.identity(acc, name = scalar_name)
            if log_scalar: tf.summary.scalar('scalar.' + scalar_name, acc)
            return acc

    def add_layer_batch_norm(self, layer_name:str, x_name:str, y_name:str, trainable:bool = True, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow:
                self._add_shadow_unit(self.add_layer_batch_norm, (layer_name, x_name, y_name, False), (1,))
            X = self.get_tensor(x_name)
            with self.scope(layer_name):
                x = tf.identity(X)
                Y = self._batch_norm(x, trainable = trainable, has_shadow = has_shadow)
            return tf.identity(Y, name = y_name)

    def add_layer_make_img(self, layer_name:str, x_name:str, y_name:str, img_shape, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow:
                self._add_shadow_unit(self.add_layer_make_img, (layer_name, x_name, y_name, img_shape), (1,))
            X = self.get_tensor(x_name)
            with self.scope(layer_name):
                x = tf.identity(X)
                assert _tensor_shape_model(x)[-1] == reduce(int.__mul__, img_shape)
                Y = tf.reshape(x, tf.concat([_tensor_shape_runtime(x)[0 : -1], img_shape], 0))
            return tf.identity(Y, name = y_name)

    def add_layer_flat_img(self, layer_name:str, x_name:str, y_name:str, img_dim:int = 3, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow:
                self._add_shadow_unit(self.add_layer_flat_img, (layer_name, x_name, y_name, img_dim), (1,))
            X = self.get_tensor(x_name)
            with self.scope(layer_name):
                x = tf.identity(X)
                img_size = reduce(int.__mul__, _tensor_shape_model(x)[-img_dim : ])
                Y = tf.reshape(x, tf.concat([_tensor_shape_runtime(x)[0 : -img_dim], [img_size]], 0))
            return tf.identity(Y, name = y_name)

    def add_layer_pool_2d(self, layer_name:str, x_name:str, y_name:str, pool_size:list,
                          step_size:list = None, padding:str = 'SAME', pool_func = tf.nn.max_pool, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow:
                self._add_shadow_unit(self.add_layer_pool_2d,
                                      (layer_name, x_name, y_name, pool_size, step_size, padding, pool_func), (1,))
            x = self.get_tensor(x_name)
            with self.scope(layer_name):
                X, shape_high = self._shape_in(x, 3)
                pool_shape = [1] + pool_size + [1]
                if step_size is None: step_size = pool_size
                strides = [1] + step_size + [1]
                Y = self._shape_out(pool_func(X, pool_shape, strides, padding), shape_high)
            return tf.identity(Y, name = y_name)

    def add_layer_full_conn(self, layer_name:str, x_name:str, y_name:str, y_size:int, act_func = None,
                            dropout_rate_name:str = None, batch_norm:bool = False,
                            trainable:bool = True, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow:
                self._add_shadow_unit(self.add_layer_full_conn,
                                      (layer_name, x_name, y_name, y_size, act_func, dropout_rate_name, batch_norm, False), (1,))
            x = self.get_tensor(x_name)
            print(x)
            input()
            with self.scope(layer_name):
                X, shape_high = self._shape_in(x, 1)
                x_size = _tensor_shape_model(X)[-1]
                w_shape = [x_size, y_size]
                b_shape = [y_size]
                if dropout_rate_name is None:
                    dropout_rate = 0
                else:
                    dropout_rate = self._get_runtime_parameter(dropout_rate_name, tf.float32, 0)
                W = self.add_variable('W', tf.random_normal(w_shape, dtype = tf.float32) / x_size,
                                      trainable = trainable, has_shadow = has_shadow)
                B = self.add_variable('B', tf.random_normal(b_shape, dtype = tf.float32) / y_size,
                                      trainable = trainable, has_shadow = has_shadow)
                Y = tf.add(tf.matmul(X, W), B, name = 'XW_B')
                if batch_norm:
                    with self.scope(_LOCAL_SCOPE_BATCH_NORM):
                        Y = self._batch_norm(Y, trainable = trainable, has_shadow = has_shadow)
                if not (act_func is None): Y = act_func(Y)
                if __tfv__ >= '1.14':
                    Y = tf.nn.dropout(Y, rate = dropout_rate, name = 'dropout')
                else:
                    Y = tf.nn.dropout(Y, keep_prob = 1 - dropout_rate, name = 'dropout')
                Y = self._shape_out(Y, shape_high)
            return tf.identity(Y, name = y_name)

    def add_layer_conv_2d(self, layer_name:str, x_name:str, y_name:str, filter_size:list, filter_num:int,
                          step_size:list = [1, 1], padding:str = 'SAME', act_func = None,
                          batch_norm:bool = False, trainable:bool = True, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow:
                self._add_shadow_unit(self.add_layer_conv_2d,
                                      (layer_name, x_name, y_name, filter_size, filter_num,
                                       step_size, padding, act_func, batch_norm, False), (1,))
            x = self.get_tensor(x_name)
            with self.scope(layer_name):
                X, shape_high = self._shape_in(x, 3)
                x_shape = _tensor_shape_model(X)
                x_channel = x_shape[-1 : ]
                f_shape = filter_size + x_channel + [filter_num]
                b_shape = [filter_num]
                strides = [1] + step_size + [1]
                F = self.add_variable('F',
                                      tf.random_normal(f_shape, dtype = tf.float32) / reduce(int.__mul__, f_shape[0 : 3]),
                                      trainable = trainable, has_shadow = has_shadow)
                B = self.add_variable('B',
                                      tf.random_normal(b_shape, dtype = tf.float32) / reduce(int.__mul__, x_shape[-3 : -1]) / filter_num,
                                      trainable = trainable, has_shadow = has_shadow)
                Y = tf.add(tf.nn.conv2d(X, F, strides, padding), B)
                if batch_norm:
                    with self.scope(_LOCAL_SCOPE_BATCH_NORM):
                        Y = self._batch_norm(Y, trainable = trainable, has_shadow = has_shadow)
                if not (act_func is None): Y = act_func(Y)
                Y = self._shape_out(Y, shape_high)
            return tf.identity(Y, name = y_name)

    def add_layer_lstm(self, layer_name:str, x_name:str, y_name:str, y_size:int,
                       trainable:bool = True, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow:
                self._add_shadow_unit(self.add_layer_lstm, (layer_name, x_name, y_name, y_size, False), (1,))
            x = self.get_tensor(x_name)
            with self.scope(layer_name) as scope:
                X, shape_high = self._shape_in(x, 1, as_seq = True)
            scope_main = scope.replace(GLOBAL_SCOPE_SHADOW, '')
            in_shadow = (scope.find(GLOBAL_SCOPE_SHADOW) == 0)
            seq_len = self._get_sequence_length(shape_high)
            initial_c, initial_h = self._get_initial_state_tensors(GLOBAL_SCOPE_INITIAL_STATE + scope_main,
                                                                  scope_main + _LOCAL_SCOPE_FINAL_STATE + '/', X, y_size)
            with tf.name_scope(None), tf.variable_scope(self._none_scope['variable_scope'],
                                                        reuse = tf.AUTO_REUSE, auxiliary_name_scope = False):
                name_scope_full = self._current_root['name_scope'] + scope + _LOCAL_SCOPE_LSTM_THROUGH_SEQ + '/'
                variable_scope_full = self._current_root['variable_scope'] + scope + _LOCAL_SCOPE_LSTM_THROUGH_SEQ + '/'
                if __tfv__ >= '1.14':
                    from tensorflow.python.ops import variables as tf_variables
                    class LSTMCell(tf.keras.layers.LSTMCell):
                        def __init__(self, units, variable_scope_full, activation='tanh', recurrent_activation='hard_sigmoid',
                                           use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                                           bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
                                           recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None,
                                           recurrent_constraint=None, bias_constraint=None, dropout=0., recurrent_dropout=0.,
                                           implementation=1, **kwargs):
                            self._variable_scope_full = variable_scope_full
                            super().__init__(units, activation, recurrent_activation, use_bias, kernel_initializer, recurrent_initializer,
                                           bias_initializer, unit_forget_bias, kernel_regularizer, recurrent_regularizer, bias_regularizer,
                                           kernel_constraint, recurrent_constraint, bias_constraint, dropout, recurrent_dropout,
                                           implementation, **kwargs)
                        def add_weight(self, name=None, shape=None, dtype=None, initializer=None, regularizer=None,
                                                 trainable=None, constraint=None, partitioner=None, use_resource=None,
                                                 synchronization=tf_variables.VariableSynchronization.AUTO,
                                                 aggregation=tf_variables.VariableAggregation.NONE, **kwargs):
                            name_full = self._variable_scope_full + name
                            kwargs['getter'] = tf.get_variable
                            weight = super().add_weight(name_full, shape, dtype, initializer, regularizer,
                                                        trainable, constraint, partitioner, use_resource,
                                                        synchronization, aggregation, **kwargs)
                            return weight
                    initial_state = [initial_h, initial_c]
                    lstm_cell = LSTMCell(y_size, variable_scope_full, name = name_scope_full + 'cell')
                    rnn_builder = tf.keras.layers.RNN(cell = lstm_cell, return_sequences = True,
                                                      return_state = True, name = name_scope_full)
                    raw_out, f_h, f_c = rnn_builder(inputs = X, initial_state = initial_state)
                    final_state = tf.nn.rnn_cell.LSTMStateTuple(f_c, f_h)
                    self._complete_variable(lstm_cell.kernel, trainable, has_shadow, False)
                    self._complete_variable(lstm_cell.bias, trainable, has_shadow, False)
                    self._complete_variable(lstm_cell.recurrent_kernel, trainable, has_shadow, False)
                else:
                    initial_state = tf.nn.rnn_cell.LSTMStateTuple(initial_c, initial_h)
                    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(y_size)
                    raw_out, final_state = tf.nn.dynamic_rnn(lstm_cell, X, initial_state = initial_state, scope = scope_full)
                    self._complete_variable(lstm_cell._kernel, trainable, has_shadow, False)
                    self._complete_variable(lstm_cell._bias, trainable, has_shadow, False)
            with self.scope(scope):
                tf.identity(raw_out, name = 'raw_out')
                Y = self.add_layer_zero_seq_tail('zero_tail', 'raw_out', 'y')
                Y = self._shape_out(Y, shape_high, as_seq = True)
                with self.scope(_LOCAL_SCOPE_FINAL_STATE):
                    final_c = tf.identity(final_state.c, name = 'c')
                    if in_shadow:
                        self._final_state_shadow.append(final_c)
                    else:
                        self._final_state.append(final_c)
                    final_h = tf.identity(final_state.h, name = 'h')
                    if in_shadow:
                        self._final_state_shadow.append(final_h)
                    else:
                        self._final_state.append(final_h)
            return tf.identity(Y, name = y_name)

    def add_layer_sequence_tail(self, layer_name:str, x_name:str, y_name:str,
                                align_to:str, low_dim:int = 1, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow:
                self._add_shadow_unit(self.add_layer_sequence_tail, (layer_name, x_name, y_name, align_to, low_dim), (1, 3))
            x = self.get_tensor(x_name)
            a = self.get_tensor(align_to)
            with self.scope(layer_name):
                X, shape_high = self._shape_in(x, low_dim, as_seq = True)
                A, a_high = self._shape_in(a, low_dim, as_seq = True)
                tail_high = tf.concat([shape_high[ : -1], a_high[-1 : ]], 0)
                Y = self._shape_out(X[ : , -tail_high[-1] : ], tail_high, as_seq = True)
            return tf.identity(Y, name = y_name)

    def add_layer_zero_seq_tail(self, layer_name:str, x_name:str, y_name:str,
                                low_dim:int = 1, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow:
                self._add_shadow_unit(self.add_layer_zero_seq_tail, (layer_name, x_name, y_name, low_dim), (1,))
            x = self.get_tensor(x_name)
            with self.scope(layer_name):
                X, shape_high = self._shape_in(x, low_dim, as_seq = True)
                sequence_length = self._get_sequence_length(shape_high)
                shape_x = _tensor_shape_runtime(X)
                shape_x_flat = shape_x[ : 2]
                R = tf.broadcast_to(tf.cast(tf.range(0, shape_x_flat[1], dtype = tf.int64), tf.float32), shape_x_flat)
                L = tf.broadcast_to(tf.cast(tf.expand_dims(sequence_length, -1), dtype = tf.float32), shape_x_flat)
                cond = R < L
                if low_dim >0:
                    cond = tf.reshape(cond, shape = tf.concat([shape_x_flat, [1] * low_dim], 0))
                    cond = tf.broadcast_to(cond, shape_x)
                Y = tf.where(cond, X, tf.zeros_like(X, dtype = X.dtype))
                Y = self._shape_out(Y, shape_high, as_seq = True)
            return tf.identity(Y, name = y_name)

    def add_layer_select(self, layer_name:str, x_name:str, y_name:str, selector_name:str, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow:
                self._add_shadow_unit(self.add_layer_select, (layer_name, x_name, y_name, selector_name), (1, 3))
            selector = self.get_tensor(selector_name)
            X = self.get_tensor(x_name)
            with self.scope(layer_name):
                x = tf.identity(X)
                s = tf.identity(selector)
                shape_x = _tensor_shape_model(x)
                dim_num = len(shape_x)
                idx_num = shape_x[-1]
                mask_unstacked = [tf.equal(s, i) for i in range(idx_num)]
                mask_stacked = tf.stack(mask_unstacked)
                mask = tf.transpose(mask_stacked, list(range(1, dim_num)) + [0])
                Y = tf.reduce_sum(tf.where(mask, x, tf.zeros_like(x, dtype = x.dtype)), axis = -1)
            return tf.identity(Y, name = y_name)

    def add_layer_duel_q(self, layer_name:str, v_name:str, a_name:str, q_name:str, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow:
                self._add_shadow_unit(self.add_layer_duel_q, (layer_name, v_name, a_name, q_name), (1, 2))
            V = self.get_tensor(v_name)
            A = self.get_tensor(a_name)
            with self.scope(layer_name):
                v, shape_high_v = self._shape_in(V, 1)
                assert v.get_shape().as_list()[-1] == 1
                a, shape_high_a = self._shape_in(A, 1)
                mean_a = tf.expand_dims(tf.reduce_mean(a, axis = -1), -1)
                q = v + a - mean_a
                q = self._shape_out(q, shape_high_v)
            return tf.identity(q, name = q_name)

    def add_queue(self, queue_name:str, capacity:int, in_name:str, out_name:str,
                  out_shape = None, dtype = None, queue_class = None, has_shadow:bool = False):
        with self.graph.as_default():
            if has_shadow:
                self._add_shadow_unit(self.add_queue, (queue_name, capacity, in_name, out_name, queue_class), (2,))
            if queue_class is None: queue_class = tf.FIFOQueue
            in_tensor = self.get_tensor(in_name)
            base_scope = self.graph.get_name_scope()
            if base_scope != '': base_scope += '/'
            with self.scope(queue_name) as scope:
                input = None
                if in_tensor is None:
                    assert (out_shape is not None) and _is_reshapable(out_shape) and (dtype is not None)
                else:
                    input = tf.identity(in_tensor)
                    if out_shape is None:
                        out_shape = _tensor_shape_model(input)
                        if not _is_reshapable(out_shape):
                            out_shape = _tensor_shape_runtime(input)
                    dtype = input.dtype
                Q = queue_class(capacity, dtypes = dtype, name = 'queue')
                dequeue = Q.dequeue(name = 'dequeue')
                out = tf.reshape(dequeue, shape = out_shape)
                queue_name = self._current_root['name_scope'] + scope[ : -1]
                self._queues[queue_name] = {'queue_obj': Q, 'pop_op': dequeue, 'root_scope': self._current_root['name_scope'],
                                            'base_scope': base_scope, 'out_shape': out_shape if isinstance(out_shape, tf.Tensor) else None}
                self._queues[queue_name]['push_op'] = self._make_push_op(queue_name, in_name if input is None else input)
            return tf.identity(out, name = out_name)

    def get_tensor(self, node, shadow:bool = False):
        with self.graph.as_default():
            prefix = GLOBAL_SCOPE_SHADOW if shadow else ''
            name_root = self._current_root['name_scope']
            variable_root = self._current_root['variable_scope']
            search_name_root = name_root + prefix
            search_variable_root = variable_root + prefix
            current_name_scope = self.graph.get_name_scope()
            if current_name_scope != '': current_name_scope += '/'
            current_variable_scope = tf.get_variable_scope().name
            if current_variable_scope != '': current_variable_scope += '/'
            search_name_scope = search_name_root + current_name_scope[len(name_root) : ]
            search_variable_scope = search_variable_root + current_variable_scope[len(variable_root) : ]
            multi_nodes = is_list(node)
            if not multi_nodes: node = [node]
            t_num = len(node)
            tensor = [None] * t_num
            for i in range(t_num):
                if isinstance(node[i], str):
                    tensor[i] = None
                    try: tensor[i] = self._get_variable_by_name(search_variable_scope + node[i] + ':0')
                    except: pass
                    if tensor[i] is None:
                        try: tensor[i] = self.graph.get_tensor_by_name(search_name_scope + node[i] + ':0')
                        except: pass
                    if tensor[i] is None:
                        try: tensor[i] = self.graph.get_tensor_by_name(search_name_scope + node[i])
                        except: pass
                    if tensor[i] is None:
                        try: tensor[i] = self.graph.get_operation_by_name(search_name_scope + node[i])
                        except: pass
                    if tensor[i] is None:
                        try: tensor[i] = self._get_variable_by_name(search_variable_root + node[i] + ':0')
                        except: pass
                    if tensor[i] is None:
                        try: tensor[i] = self.graph.get_tensor_by_name(search_name_root + node[i] + ':0')
                        except: pass
                    if tensor[i] is None:
                        try: tensor[i] = self.graph.get_tensor_by_name(search_name_root + node[i])
                        except: pass
                    if tensor[i] is None:
                        try: tensor[i] = self.graph.get_operation_by_name(search_name_root + node[i])
                        except: pass
                else: tensor[i] = node[i]
            return tensor if multi_nodes else tensor[0]

    def get_pull_essential_trainer(self):
        if self._trainer is None: return None
        else: return self._batch_norm_paras_updaters['tensors'] + [self._trainer]

    def get_pull_essential_final_state(self, shadow:bool = False):
        fetches = self._final_state_shadow if shadow else self._final_state
        return fetches

    def get_pull_essential_queue(self, queue_name):
        if queue_name is None: return None
        queues = queue_name if is_list(queue_name) else [queue_name]
        push_ops = []
        pop_ops = []
        for q in queues:
            try:
                push_op = self._queues[q]['push_op']
                pop_op = self._queues[q]['pop_op']
            except KeyError:
                push_ops.append(None)
                pop_ops.append(None)
                continue
            if isinstance(push_op, str):
                self._pub_mutex.lock()
                push_op = self._queues[q]['push_op']
                if isinstance(push_op, str):
                    push_op = self._make_push_op(q, push_op)
                    self._queues[q]['push_op'] = push_op
                self._pub_mutex.unlock()
                if isinstance(push_op, str): push_op = None
            push_ops.append(push_op)
            pop_ops.append(pop_op)
        if is_list(queue_name): return push_ops, pop_ops
        else: return push_ops[0], pop_ops[0]

    def get_pull_essential_summary(self):
        with self.graph.as_default():
            if self._merge_summary is None:
                self._pub_mutex.lock()
                if self._merge_summary is None: self._merge_summary = tf.summary.merge_all()
                self._pub_mutex.unlock()
            return self._merge_summary

    def get_queue_essntial_rnn_state(self):
        return self._rnn_state_queues, self._final_state

    def set_trainer(self, optimizer, loss_name:str, colocate_gradients_with_ops:bool = False):
        if self._trainer is not None:
            raise ValueError('Trainer can be set only once.')
        with self.graph.as_default():
            learning_rate = self._get_runtime_parameter('learning_rate', tf.float32, None)
            loss = self.get_tensor(loss_name)
            tmp_vars = set(tf.all_variables())
            self._trainer = optimizer(learning_rate).minimize(loss, colocate_gradients_with_ops = colocate_gradients_with_ops)
            trainer_vars = set(tf.all_variables()) - tmp_vars
            self._initializer_trainer = tf.variables_initializer(trainer_vars, name = 'init_trainer')
            for var in trainer_vars: self._complete_variable(var, trainable = True, has_shadow = False, for_trainer = True)
            return self._trainer

    def logs_path(self, path:str = None, clear:bool = False, make_tensorbord_runner:bool = False):
        if path is not None: self._logs_path = path
        if clear:
            try: rmtree(self._logs_path)
            except FileNotFoundError: pass
        if make_tensorbord_runner: self.make_tensorbord_runner()
        return self._logs_path

    def make_tensorbord_runner(self):
        return func_mtr(self._logs_path)

    def print_graph(self):
        for queue_name in self._queues.keys():
            self.get_pull_essential_queue(queue_name)
        with self.graph.as_default():
            graph_path = self._logs_path + '/graph'
            try: rmtree(graph_path)
            except FileNotFoundError: pass
            writer = tf.summary.FileWriter(graph_path, self.graph)
            writer.close()

    def print_summary(self, summary, step, series_name = None, h_sess = None, multi_thread:bool = False):
        multi_ret = self._solve_none_and_multi_sess(self.print_summary, (summary, step, series_name, h_sess, multi_thread),
                                                                            3, (0, 1, 2), multi_thread = multi_thread)
        if is_list(multi_ret): return multi_ret
        if multi_ret is not None: h_sess = multi_ret
        sess_str = '' if len(self._sess) < 2 else ('%d' % h_sess)
        if series_name is None: series_name = ''
        sub_path = '_'.join([s for s in [sess_str, series_name] if s != ''])
        series_path = '%s/%s' % (self._logs_path, sub_path)
        writer = tf.summary.FileWriter(series_path)
        writer.add_summary(summary, global_step = step)
        writer.close()

    def session_new(self, num:int = None, target = None, initialize:bool = True, inter_op_threads:int = 0):
        multi = num is not None
        if num is None: num = 1
        if num < 1: return None
        if target is None: target = ''
        tf_config = tf.ConfigProto(inter_op_parallelism_threads = inter_op_threads)
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True
        tf_config.log_device_placement = False
        h_sess = []
        self._pub_mutex.lock()
        if self._initializer_global is None:
            with self.graph.as_default():
                self._initializer_global = tf.global_variables_initializer()
        for i in range(num):
            if is_list(target): tar = target[i]
            else: tar = target
            self._sess_head += 1
            self._sess[self._sess_head] = tf.Session(graph = self.graph, config = tf_config, target = tar)
            self._sess_mutex[self._sess_head] = BasicLock()
            if initialize: self.session_reset_global(self._sess_head)
            h_sess.append(self._sess_head)
        ret = (h_sess if multi else self._sess_head)
        self._pub_mutex.unlock()
        return ret

    def session_close(self, h_sess = None):
        multi_ret = self._solve_none_and_multi_sess(self.session_close, (h_sess,), 0)
        if is_list(multi_ret): return multi_ret
        if multi_ret is not None: h_sess = multi_ret
        self._pub_mutex.lock()
        sess = self._get_session(h_sess)
        if sess is None: ret = False
        else:
            sess.close()
            self._sess.pop(h_sess)
            mutex = self._sess_mutex.pop(h_sess)
            mutex.unlock()
            ret = True
        self._pub_mutex.unlock()
        return ret

    def session_reset_global(self, h_sess = None, multi_thread:bool = False):
        multi_ret = self._solve_none_and_multi_sess(self.session_reset_global,
                                                    (h_sess, multi_thread), 0, multi_thread = multi_thread)
        if is_list(multi_ret): return multi_ret
        if multi_ret is not None: h_sess = multi_ret
        sess = self._get_session(h_sess)
        if sess is None: return False
        mutex = self._sess_mutex[h_sess]
        ret = True
        mutex.lock()
        try:
            sess.run(self._initializer_global)
            copy_tensors = list()
            for v in self._var_dict.values(): copy_tensors += v['copy']
            if len(copy_tensors) > 0:
                with self.graph.as_default():
                    sess.run(copy_tensors)
            shadow_assigner_tensors = self._shadow_assigners['tensors']
            if len(shadow_assigner_tensors) > 0:
                with self.graph.as_default():
                    sess.run(shadow_assigner_tensors)
        except (CancelledError, RuntimeError, InvalidArgumentError): ret = False
        mutex.unlock()
        return ret

    def session_reset_trainer(self, h_sess = None, multi_thread:bool = False):
        multi_ret = self._solve_none_and_multi_sess(self.session_reset_trainer,
                                                    (h_sess, multi_thread), 0, multi_thread = multi_thread)
        if is_list(multi_ret): return multi_ret
        if multi_ret is not None: h_sess = multi_ret
        sess = self._get_session(h_sess)
        if sess is None: return False
        ret = True
        try: sess.run(self._initializer_trainer)
        except (CancelledError, RuntimeError, InvalidArgumentError): ret = False
        return ret

    def session_pull(self, node, h_sess = None, feed_dict:dict = None, dropout_rate_dict:dict = None,
                             learning_rate:float = 0, queued_item_shape_dict:dict = None, sequence_length:array = None,
                             initial_state:tuple = None, rnn_state_queue_mode:str = None,
                             norm_in_batch:bool = False, multi_thread:bool = False):
        multi_ret = self._solve_none_and_multi_sess(self.session_pull, (node, h_sess, feed_dict,
                                                                            dropout_rate_dict, learning_rate, queued_item_shape_dict,
                                                                            sequence_length, initial_state, rnn_state_queue_mode,
                                                                            norm_in_batch, multi_thread),
                                                                            1, tuple(range(2, 10)), multi_thread = multi_thread)
        if is_list(multi_ret): return multi_ret
        if multi_ret is not None: h_sess = multi_ret
        sess = self._get_session(h_sess)
        if sess is None or node is None: return None
        if rnn_state_queue_mode is None: rnn_state_queue_mode = 'keep'
        push_rnn_state_queue = (rnn_state_queue_mode != 'keep')
        pop_rnn_state_queue = (rnn_state_queue_mode == 'cyclic') or (rnn_state_queue_mode == 'replace')
        feed_initial_state = (rnn_state_queue_mode != 'cyclic')
        with self.graph.as_default():
            tensor = self._get_tensor_from_none_root(node)
            if push_rnn_state_queue:
                queues = self._rnn_state_queues
                if len(queues) > 0:
                    push_ops, pop_ops = self.get_pull_essential_queue(queues)
                    if not is_list(tensor): tensor = [tensor]
                    tensor = tensor + push_ops
                    if pop_rnn_state_queue: tensor = tensor + pop_ops
            fd = self._make_feed(feed_dict, dropout_rate_dict, learning_rate, queued_item_shape_dict,
                                             sequence_length, initial_state, feed_initial_state, norm_in_batch)
            try: value = sess.run(tensor, fd)
            except (CancelledError, RuntimeError, InvalidArgumentError): return None
            if is_list(node):
                return tuple(value[ : len(node)])
            else:
                return (value[0] if is_list(value) else value)

    def session_train(self, h_sess=None, feed_dict: dict = None, dropout_rate_dict: dict = None,
                      learning_rate: float = 0, queued_item_shape_dict: dict = None, sequence_length: array = None,
                      initial_state: tuple = None, multi_thread: bool = False):
        essential = self.get_pull_essential_trainer()
        return self.session_pull(essential, h_sess = h_sess, feed_dict = feed_dict, dropout_rate_dict = dropout_rate_dict,
                                 learning_rate = learning_rate, queued_item_shape_dict = queued_item_shape_dict,
                                 sequence_length = sequence_length, initial_state = initial_state,
                                 norm_in_batch = True, multi_thread = multi_thread)

    def session_final_state(self, h_sess = None, feed_dict:dict = None, dropout_rate_dict:dict = None,
                            queued_item_shape_dict:dict = None, sequence_length:array = None,
                            initial_state:tuple = None, rnn_state_queue_mode:str = None, shadow:bool = False,
                            norm_in_batch:bool = False, multi_thread:bool = False):
        essential = self.get_pull_essential_final_state(shadow)
        return self.session_pull(essential, h_sess = h_sess, feed_dict = feed_dict, dropout_rate_dict = dropout_rate_dict,
                                        queued_item_shape_dict = queued_item_shape_dict, sequence_length = sequence_length,
                                        initial_state = initial_state, rnn_state_queue_mode = rnn_state_queue_mode,
                                        norm_in_batch = norm_in_batch, multi_thread = multi_thread)

    def session_push(self, queue_name, h_sess = None, feed_dict:dict = None, dropout_rate_dict:dict = None,
                            queued_item_shape_dict:dict = None, sequence_length:array = None,
                            initial_state:tuple = None, rnn_state_queue_mode:str = None,
                            norm_in_batch:bool = False, multi_thread:bool = False):
        push_ops, _ = self.get_pull_essential_queue(queue_name)
        return self.session_pull(push_ops, h_sess = h_sess, feed_dict = feed_dict, dropout_rate_dict = dropout_rate_dict,
                                        queued_item_shape_dict = queued_item_shape_dict, sequence_length = sequence_length,
                                        initial_state = initial_state, rnn_state_queue_mode = rnn_state_queue_mode,
                                        norm_in_batch = norm_in_batch, multi_thread = multi_thread)

    def session_push_rnn_state(self, rnn_state:tuple = None, h_sess = None, multi_thread:bool = False):
        queues, feed_keys = self.get_queue_essntial_rnn_state()
        if rnn_state is None:
            rnn_state = []
            for tensor in feed_keys:
                zero_state = [0.0] * _tensor_shape_model(tensor)[-1]
                rnn_state.append(array([zero_state], dtype = numpy.float32))
            rnn_state = tuple(rnn_state)
        multi_ret = self._solve_none_and_multi_sess(self.session_push_rnn_state, (rnn_state, h_sess, multi_thread),
                                                                             1, (0,), multi_thread = multi_thread)
        if is_list(multi_ret): return multi_ret
        if multi_ret is not None: h_sess = multi_ret
        fd = {k: v for k, v in zip(feed_keys, rnn_state)}
        return self.session_push(queues, h_sess = h_sess, feed_dict = fd, multi_thread = multi_thread)

    def session_pop(self, queue_name, h_sess = None, multi_thread:bool = False):
        _, pop_ops = self.get_pull_essential_queue(queue_name)
        return self.session_pull(pop_ops, h_sess = h_sess, multi_thread = multi_thread)

    def session_pop_rnn_state(self, h_sess = None, multi_thread:bool = False):
        queues, _ = self.get_queue_essntial_rnn_state()
        return self.session_pop(queues, h_sess = h_sess, multi_thread = multi_thread)

    def session_clone_variable(self, src_name:str, dst_name:str, h_sess = None, multi_thread:bool = False):
        multi_ret = self._solve_none_and_multi_sess(self.session_clone_variable,
                                                    (src_name, dst_name, h_sess, multi_thread),
                                                    2, [0, 1], multi_thread = multi_thread)
        if is_list(multi_ret): return multi_ret
        if multi_ret is not None: h_sess = multi_ret
        sess = self._get_session(h_sess)
        if sess is None: return None
        with self.graph.as_default():
            clone_name = ('clone_' + src_name + '_to_' + dst_name).replace('/', '~')
            self._pub_mutex.lock()
            tensor = self.get_tensor(clone_name)
            if tensor is None:
                src = self.get_tensor(src_name)
                dst = self.get_tensor(dst_name)
                tensor = tf.assign(dst, src, name = clone_name)
            self._pub_mutex.unlock()
            try: ret = sess.run(tensor)
            except (CancelledError, RuntimeError, InvalidArgumentError): return None
            return ret

    def session_update_variable_copies(self, h_sess = None, multi_thread:bool = False):
        multi_ret = self._solve_none_and_multi_sess(self.session_update_variable_copies,
                                                    (h_sess, multi_thread), 0, multi_thread = multi_thread)
        if is_list(multi_ret): return multi_ret
        if multi_ret is not None: h_sess = multi_ret
        sess = self._get_session(h_sess)
        if sess is None: return None
        copy_tensors = list()
        for v in self._var_dict.values(): copy_tensors += v['copy']
        if len(copy_tensors) == 0: return None
        with self.graph.as_default():
            try: ret = tuple(sess.run(copy_tensors))
            except (CancelledError, RuntimeError, InvalidArgumentError): return None
            return ret

    def session_update_shadow(self, h_sess = None, multi_thread:bool = False):
        multi_ret = self._solve_none_and_multi_sess(self.session_update_shadow,
                                                    (h_sess, multi_thread), 0, multi_thread = multi_thread)
        if is_list(multi_ret): return multi_ret
        if multi_ret is not None: h_sess = multi_ret
        sess = self._get_session(h_sess)
        if sess is None: return None
        shadow_assigner_tensors = self._shadow_assigners['tensors']
        if len(shadow_assigner_tensors) == 0: return None
        with self.graph.as_default():
            try: ret = tuple(sess.run(shadow_assigner_tensors))
            except (CancelledError, RuntimeError, InvalidArgumentError): return None
            return ret

    def session_export_variables(self, h_sess = None, var_filters = None, multi_thread:bool = False):
        multi_ret = self._solve_none_and_multi_sess(self.session_export_variables,
                                                    (h_sess, var_filters, multi_thread), 0, multi_thread = multi_thread)
        if is_list(multi_ret): return multi_ret
        if multi_ret is not None: h_sess = multi_ret
        sess = self._get_session(h_sess)
        if sess is None: return {}
        if hasattr(var_filters, '__call__'): var_filters = [var_filters]
        with self.graph.as_default():
            pairs = [(k, v['var']) for k, v in self._var_dict.items() if
                     (var_filters is None or reduce(bool.__and__, [var_filter(k, v) for var_filter in var_filters]))]
            try: ret = {name: sess.run(var) for name, var in pairs}
            except (CancelledError, RuntimeError, InvalidArgumentError): return {}
            return ret

    def session_import_variables(self, value_dict:dict, h_sess = None, var_filters = None, multi_thread:bool = False):
        multi_ret = self._solve_none_and_multi_sess(self.session_import_variables,
                                                    (value_dict, h_sess, var_filters, multi_thread), 1, [0], multi_thread = multi_thread)
        if is_list(multi_ret): return multi_ret
        if multi_ret is not None: h_sess = multi_ret
        sess = self._get_session(h_sess)
        if sess is None: return {}
        if hasattr(var_filters, '__call__'): var_filters = [var_filters]
        with self.graph.as_default():
            loaders = []
            fd = {}
            imported = {}
            for name, value in value_dict.items():
                for k, v in self._var_dict.items():
                    if (k == name or k == name + ':0') and \
                        (var_filters is None or reduce(bool.__and__, [var_filter(k, v) for var_filter in var_filters])):
                        loaders += v['loaders']
                        value_as_array = array(value)
                        fd[v['src']] = value_as_array
                        imported[name] = value_as_array
                        break
            try: sess.run(loaders, feed_dict = fd)
            except (CancelledError, RuntimeError, InvalidArgumentError): return {}
            return imported

    def session_export_variables_to_file(self, filename:str, h_sess = None, var_filters = None, multi_thread:bool = False):
        multi_ret = self._solve_none_and_multi_sess(self.session_export_variables_to_file,
                                                    (filename, h_sess, var_filters, multi_thread), 1, [0], multi_thread = multi_thread)
        if is_list(multi_ret): return multi_ret
        if multi_ret is not None: h_sess = multi_ret
        value_dict = self.session_export_variables(h_sess, var_filters)
        path = get_file_path_from_name(filename)
        try: file = open(filename, 'wb')
        except FileNotFoundError:
            try:
                makedirs(path, 0o777)
                file = open(filename, 'wb')
            except: return {}
        except: return {}
        dump_pack(value_dict, file)
        file.close()
        return value_dict

    def  session_import_variables_from_file(self, filename:str, h_sess = None, var_filters = None, multi_thread:bool = False):
        multi_ret = self._solve_none_and_multi_sess(self.session_import_variables_from_file,
                                                    (filename, h_sess, var_filters, multi_thread), 1, [0], multi_thread = multi_thread)
        if is_list(multi_ret): return multi_ret
        if multi_ret is not None: h_sess = multi_ret
        try: file = open(filename, 'rb')
        except: return {}
        value_dict = load_pack(file)
        file.close()
        return self.session_import_variables(value_dict, h_sess, var_filters)


class ScalarLogger:
    def __init__(self, name, logs_path, dtype = tf.float32):
        self._network = TF_Neural_Network()
        self._scalar = self._network.add_placeholder('scalar' + name, dtype = dtype)
        with self._network.graph.as_default(): tf.summary.scalar(name, self._scalar)
        self._network.logs_path(logs_path + '/' + name)
        self._network.session_new()

    def __del__(self):
        self._network.session_close()

    def log(self, value, step, series_name = None):
        nn = self._network
        summary_essential = nn.get_pull_essential_summary()
        summary = nn.session_pull(summary_essential, feed_dict = {self._scalar: value})
        nn.print_summary(summary, step, series_name = series_name)

    def clear(self):
        self._network.logs_path(clear = True)
