__all__ = ['PickSelector', 'Experience_Replay']
# 这是老师的代码，原原本本没有改过，replay和ddpg被dsq和lyk改过，td3被dsq改过

from .Funcs import *
from .Locker import BasicLock

from ctypes import *
from _pickle import load as load_pack
from _pickle import dump as dump_pack
from os import makedirs
from numpy import array
from functools import reduce
from copy import deepcopy as COPY


file_path = get_file_path_from_name(__file__)
_er_kernel = CDLL(file_path + '/../cpp/bin/er_kernel.dll')

LH_EPI_T = c_int64
LH_REC_T = c_uint64
PTR = c_uint64
H_REC_T = c_int32
BATCH_SIZE_T = c_uint32
H_PS_T = c_uint32
ACTION_T = c_uint16
STATE_SIZE_T = c_uint16
STATE_T = c_float
REWARD_T = c_float
PRIORITY_T = c_float
SEQ_LEN_T = H_REC_T

_ex_rp_new = _er_kernel.ex_rp_new
_ex_rp_new.restype = PTR
_ex_rp_del = _er_kernel.ex_rp_del
_ex_rp_clear = _er_kernel.ex_rp_clear
_ex_rp_serialize_size = _er_kernel.ex_rp_serialize_size
_ex_rp_serialize_size.restype = c_uint64
_ex_rp_serialize = _er_kernel.ex_rp_serialize
_ex_rp_unserialize = _er_kernel.ex_rp_unserialize
_ex_rp_get_pick_policy = _er_kernel.ex_rp_get_pick_policy
_ex_rp_new_episode = _er_kernel.ex_rp_new_episode
_ex_rp_new_episode.restype = LH_EPI_T
_ex_rp_record = _er_kernel.ex_rp_record
_ex_rp_record.restype = LH_EPI_T
_ex_rp_get_random_batch = _er_kernel.ex_rp_get_random_batch
_ex_rp_get_random_batch.restype = c_int32
_ex_rp_set_elimination_policy = _er_kernel.ex_rp_set_elimination_policy
_ex_rp_set_pick_priority = _er_kernel.ex_rp_set_pick_priority
_ex_rp_reset_pick_priorities = _er_kernel.ex_rp_reset_pick_priorities
_ex_rp_del_pick_selector = _er_kernel.ex_rp_del_pick_selector

class PickSelector:
    @staticmethod
    def greedy_rb_tree(kernel):
        return H_PS_T(_er_kernel.ex_rp_new_pick_selector_greedy_rb_tree(kernel))

    @staticmethod
    def greedy_bin_heap(kernel):
        return H_PS_T(_er_kernel.ex_rp_new_pick_selector_greedy_bin_heap(kernel))


class Experience_Replay:
    def __init__(self, max_record_num:int = 0, pick_len = None, allow_short_seq = None):
        if pick_len is None: pick_len = -1
        if allow_short_seq is None: allow_short_seq = -1
        self._kernel = PTR(_ex_rp_new(LH_REC_T(max_record_num),
                                      SEQ_LEN_T(pick_len), c_int32(allow_short_seq)))
        self._mutex = BasicLock()
        self._state_shape = None

    def __del__(self):
        self._mutex.unlock()
        _ex_rp_del(self._kernel)

    def get_pick_policy(self):
        pick_len = SEQ_LEN_T()
        allow_short_seq = c_bool()
        _ex_rp_get_pick_policy(self._kernel, byref(pick_len), byref(allow_short_seq))
        return pick_len.value, allow_short_seq.value

    def set_elimination_policy(self, second_chance_elimination):
        second_chance_elimination = int(second_chance_elimination)
        _ex_rp_set_elimination_policy(self._kernel, c_int32(second_chance_elimination))

    def clear(self):
        self._mutex.lock()
        _ex_rp_clear(self._kernel)
        self._state_shape = None
        self._mutex.unlock()

    def new_episode(self):
        self._mutex.lock()
        h_epi = _ex_rp_new_episode(self._kernel)
        self._mutex.unlock()
        return h_epi

    def new_pick_selector(self, pick_selector_class):
        self._mutex.lock()
        h_ps = pick_selector_class(self._kernel)
        self._mutex.unlock()
        return h_ps

    def del_pick_selector(self, h_ps):
        self._mutex.lock()
        _ex_rp_del_pick_selector(self._kernel, h_ps)
        self._mutex.unlock()

    def record(self, h_epi, state, action, reward = None, final_state = None):
        if is_list(h_epi):
            ep_num = len(h_epi)
            ret = [None] * ep_num
            for i in range(ep_num):
                s = state[i] if is_list(state) else state
                a = action[i] if is_list(action) else action
                r = reward[i] if is_list(reward) else reward
                fs = final_state[i] if is_list(final_state) else final_state
                ret[i] = self.record(h_epi[i], s, a, r, fs)
            return ret
        if h_epi is None: h_epi = -1
        if reward is None: reward = 0
        if final_state is None: final_state = 0
        else:
            final_state = array(final_state).reshape([-1])
            final_state = (STATE_T * len(final_state))(*final_state)
            final_state = byref(final_state)
        state = array(state)
        self._mutex.lock()
        if self._state_shape is None: self._state_shape = list(state.shape)
        state_size = reduce(int.__mul__, self._state_shape)
        state = state.reshape([-1])
        state = (STATE_T * state_size)(*state)
        h_epi = _ex_rp_record(self._kernel, LH_EPI_T(h_epi), state, ACTION_T(action),
                              REWARD_T(reward), final_state, STATE_SIZE_T(state_size))
        self._mutex.unlock()
        return h_epi

    def get_random_batch(self, batch_size:int, valid_sample_rate:float = 0, h_ps = None):
        if h_ps is None: h_ps = H_PS_T(0)
        self._mutex.lock()
        if self._state_shape is None:
            self._mutex.unlock()
            return None
        pick_len, _ = self.get_pick_policy()
        state_size = reduce(int.__mul__, self._state_shape)
        state = (STATE_T * (batch_size * pick_len * state_size))(0)
        state_ = (STATE_T * (batch_size * pick_len * state_size))(0)
        action = (ACTION_T * (batch_size * pick_len))(0)
        reward = (REWARD_T * (batch_size * pick_len))(0)
        seq_len = (SEQ_LEN_T * batch_size)(0)
        seq_len_ = (SEQ_LEN_T * batch_size)(0)
        pick_epi = (LH_EPI_T * batch_size)(0)
        pick_pos = (H_REC_T * batch_size)(0)
        ret = _ex_rp_get_random_batch(self._kernel, h_ps, BATCH_SIZE_T(batch_size),
                                      c_float(valid_sample_rate), pick_epi, pick_pos, state, action, reward, state_, seq_len, seq_len_)
        self._mutex.unlock()
        if ret == 0: return None
        state = array(state).reshape([batch_size, pick_len] + self._state_shape)
        state_ = array(state_).reshape([batch_size, pick_len] + self._state_shape)
        action = array(action).reshape([batch_size, pick_len])
        reward = array(reward).reshape([batch_size, pick_len])
        seq_len = array(seq_len)
        seq_len_ = array(seq_len_)
        pick_epi = array(pick_epi)
        pick_pos = array(pick_pos)
        return [state, action, reward, state_, seq_len, seq_len_, pick_epi, pick_pos]

    def set_priority(self, h_ps, pick_epi, pick_pos, priority:array):
        batch_size = len(pick_epi)
        pick_epi = pick_epi.ctypes.data_as(c_char_p)
        pick_pos = pick_pos.ctypes.data_as(c_char_p)
        priority = priority.ctypes.data_as(c_char_p)
        self._mutex.lock()
        _ex_rp_set_pick_priority(self._kernel, h_ps, pick_epi, pick_pos,
                                 priority, BATCH_SIZE_T(batch_size))
        self._mutex.unlock()

    def reset_priorities(self, h_ps):
        self._mutex.lock()
        _ex_rp_reset_pick_priorities(self._kernel, h_ps)
        self._mutex.unlock()

    def export_data(self):
        self._mutex.lock()
        serie_size = _ex_rp_serialize_size(self._kernel)
        serie = (c_char * serie_size)()
        _ex_rp_serialize(self._kernel, byref(serie))
        serie = tuple(serie)
        pack = (COPY(self._state_shape), serie_size, serie)
        self._mutex.unlock()
        return pack

    def import_data(self, pack):
        self._mutex.lock()
        self._state_shape = COPY(pack[0])
        serie_size = pack[1]
        serie = (c_char * serie_size)(*(pack[2]))
        _ex_rp_unserialize(self._kernel, byref(serie))
        self._mutex.unlock()
        return True

    def export_experience_to_file(self, filename:str):
        path = get_file_path_from_name(filename)
        try: file = open(filename, 'wb')
        except FileNotFoundError:
            try:
                makedirs(path, 0o777)
                file = open(filename, 'wb')
            except: return None
        except: return None
        pack = self.export_data()
        dump_pack(pack, file)
        file.close()
        return pack

    def import_experience_from_file(self, filename:str):
        try: file = open(filename, 'rb')
        except: return None
        pack = load_pack(file)
        file.close()
        if self.import_data(pack): return pack
        else: return None