__all__ = ['Actor']


from .__import__ import is_list
from .__import__ import get_file_path_from_name
from .__import__ import GroupLock
from .__import__ import Experience_Replay as ExpRep
from .__import__ import make_tensorbord_runner as func_mtr
from .Server import *

from os import makedirs
from _pickle import dump as dump_pack
from _pickle import load as load_pack
from numpy import array
from copy import deepcopy as COPY
from threading import Thread


class Actor:

    from .Config import Config
    from .Server.Flags import FLAG_MODEL_CLOSED
    from .Server.Flags import FLAG_NO_BATCH
    from .Server.Flags import FLAG_NORMAL
    from .Server.Flags import FLAG_SHADOW_UPD
    from .Server.Flags import FLAG_FULL_IMPORTED
    from .Server.Flags import FLAG_PARTIAL_IMPORTED

    @staticmethod
    def PASS_THROUGH(act_cand, trained_step, act_num):
        return act_cand

    @staticmethod
    def EPS_GREEDY_CHOOSE(act_cand, trained_step, act_num, eps_init=None,
                          eps_stable=None, eps_drop_at=None, eps_stable_at=None):
        DEFAULT_EPS_INIT = 0.95
        DEFAULT_EPS_STABLE = 0.02
        DEFAULT_EPS_DROP_AT = 3000
        DEFAULT_EPS_STABLE_AT = 100000
        if eps_init is None: eps_init = DEFAULT_EPS_INIT
        if eps_stable is None: eps_stable = DEFAULT_EPS_STABLE
        if eps_drop_at is None: eps_drop_at = DEFAULT_EPS_DROP_AT
        if eps_stable_at is None: eps_stable_at = DEFAULT_EPS_STABLE_AT
        eps = eps_init
        if trained_step >= eps_stable_at:
            eps = eps_stable
        else:
            step_delta = trained_step - eps_drop_at
            if step_delta > 0: eps = eps_init + step_delta * (eps_stable - eps_init) / (eps_stable_at - eps_drop_at)
        print("==========================！@#￥    trained_step       @#￥=", trained_step, eps)

        from numpy import random as __random
        if __random.uniform() > eps:
            return act_cand
        else:
            return __random.choice(range(act_num))

    def __init__(self, config:Config, mod_num:int = 1, cluster_config:dict = None):
        self.serv = build_server(mod_num = mod_num, config = config, cluster_config = cluster_config)
        if self.is_master():
            self.exp_rep = ExpRep(config.experience_size, config.pick_len, config.allow_short_seq)
            if config.pick_selector_class is None:
                self.h_ps = None
                self.exp_rep.set_elimination_policy(second_chance_elimination = False)
            else:
                self.h_ps = [self.exp_rep.new_pick_selector(config.pick_selector_class) for _ in range(mod_num)]
                self.exp_rep.set_elimination_policy(second_chance_elimination = True)
            self.episode_temp = {}
            self.runtime_lock = GroupLock(['run', 'save', 'load', 'clear'])
            self.master = build_master(serv = self.serv, exp_rep = self.exp_rep, h_ps = self.h_ps,
                                       mod_num = mod_num, cluster_config = cluster_config)

    def _all_models(self):
        h_mod = list(range(self.model_num()))
        if len(h_mod) == 1:
            return [0]
        else:
            return h_mod

    def _thread_func_model_train(self, idx, learning_rate, dropout_rate_dict, h_mod, ret):
        lr = learning_rate[idx] if is_list(learning_rate) else learning_rate
        drd = dropout_rate_dict[idx] if is_list(dropout_rate_dict) else dropout_rate_dict
        m = h_mod[idx] if is_list(h_mod) else h_mod
        r = self.model_train(lr, drd, m)
        ret[idx] = r

    def _thread_func_episode_act(self, idx, h_epi, state, h_mod, extra_func_args,
                                 choose_func, ret_h_epi, ret_action):
        s = state[idx] if is_list(state) else state
        st = h_mod[idx] if is_list(h_mod) else h_mod
        efg = extra_func_args[idx] if is_list(extra_func_args) else extra_func_args
        cf = choose_func[idx] if is_list(choose_func) else choose_func
        t, a = self.episode_act(h_epi[idx], s, st, efg, cf)
        ret_h_epi[idx] = t
        ret_action[idx] = a

    def _print_graphs(self, graph_root:str, make_tensorbord_runner:bool):
        threads = []
        try:
            if self.is_master():
                self.master.wait_all_models_settled()
            threads += self.serv.print_graphs_threads(graph_root)
            if make_tensorbord_runner: func_mtr(graph_root)
        except: pass
        for t in threads:
            t.join()

    def stand_by(self):
        self.serv.stand_by()
        self.serv.print_log('Server is standing by.', 1)

    def close(self):
        if self.is_master():
            close_master(self.master)
            self.runtime_clear()
        close_server(self.serv)

    def is_master(self):
        return self.serv.is_master()

    def print_graphs_thread(self, graph_root:str, make_tensorbord_runner:bool = False):
        t = Thread(target = self._print_graphs, args = [graph_root, make_tensorbord_runner], daemon = True)
        t.start()
        return t

    def model_num(self):
        if not self.is_master(): return -1
        return self.master.model_num()

    def model_reset(self, h_mod = None):
        if not self.is_master(): return
        if h_mod is None: h_mod = self._all_models()
        multi_mod = is_list(h_mod)
        if not multi_mod: h_mod = [h_mod]
        mod_num = len(h_mod)
        for i in range(mod_num):
            mod = self.master.get_model_monitor(h_mod[i])
            if mod is not None: request_reset(mod)
        self.runtime_lock.lock('run')
        for m in h_mod:
            if self.h_ps is not None: self.exp_rep.reset_priorities(self.h_ps[m])
        self.runtime_lock.unlock('run')

    def model_train(self, learning_rate:float, dropout_rate_dict:dict = None, h_mod = None):
        if not self.is_master(): return
        if h_mod is None: h_mod = self._all_models()
        if is_list(h_mod):
            mod_num = len(h_mod)
            ret = [None] * mod_num
            if mod_num > 1:
                threads = [Thread(daemon = True, target = self._thread_func_model_train,
                                            args = (idx, learning_rate, dropout_rate_dict, h_mod, ret))
                           for idx in range(mod_num)]
                for t in threads: t.start()
                for t in threads: t.join()
            else:
                self._thread_func_model_train(0, learning_rate, dropout_rate_dict, h_mod, ret)
            return ret
        mod = self.master.get_model_monitor(h_mod)
        if mod is None: return Actor.FLAG_MODEL_CLOSED
        train_ret = request_train(mod_mnt = mod, learning_rate = learning_rate, dropout_rate_dict = dropout_rate_dict)
        if train_ret is None:
            return Actor.FLAG_MODEL_CLOSED
        if train_ret == Actor.FLAG_NO_BATCH:
            return Actor.FLAG_NO_BATCH
        batch_ticket, td_error, flag, trained_step = train_ret
        pick_epi, pick_pos = self.master.get_batch_info(batch_ticket)
        self.runtime_lock.lock('run')
        if self.h_ps is not None:
            self.exp_rep.set_priority(self.h_ps[h_mod], pick_epi, pick_pos, td_error)
        self.runtime_lock.unlock('run')
        return (flag, trained_step)

    def episode_act(self, h_epi, state, h_mod = None, extra_func_args:tuple = None, choose_func = None):
        if not self.is_master(): return
        if extra_func_args is None: extra_func_args = ()
        if choose_func is None: choose_func = Actor.EPS_GREEDY_CHOOSE
        if h_mod is None: h_mod = self._all_models()
        if is_list(h_epi):
            ep_num = len(h_epi)
            ret_h_epi = [None] * ep_num
            ret_action = [None] * ep_num
            if ep_num > 1:
                threads = [Thread(daemon = True, target = self._thread_func_episode_act,
                                            args = (idx, h_epi, state, h_mod, extra_func_args,
                                                    choose_func, ret_h_epi, ret_action))
                                for idx in range(ep_num)]
                for t in threads: t.start()
                for t in threads: t.join()
            else:
                self._thread_func_episode_act(0, h_epi, state, h_mod, extra_func_args,
                                              choose_func, ret_h_epi, ret_action)
            return (ret_h_epi, ret_action)
        self.runtime_lock.lock('run')
        if is_list(h_mod): h_mod = h_mod[0]
        mod = self.master.get_model_monitor(h_mod)
        if mod is None: return Actor.FLAG_MODEL_CLOSED
        state = array(state)
        if h_epi is None: initial_state = None
        else: initial_state = self.episode_temp[h_epi]['rnn_state']
        act_ret = request_act(mod_mnt = mod, state = state, initial_state = initial_state)
        if act_ret is None: return Actor.FLAG_MODEL_CLOSED
        act_cand, final_state, trained_step = act_ret
        action = choose_func(act_cand, trained_step, self.serv.config.act_num, *extra_func_args)
        if h_epi is None:
            h_epi = self.exp_rep.new_episode()
            self.episode_temp[h_epi] = dict()
        temp = self.episode_temp[h_epi]
        temp['state'] = state
        temp['action'] = action
        temp['rnn_state'] = final_state
        self.runtime_lock.unlock('run')
        return (h_epi, action)

    def episode_feedback(self, h_epi, reward = None, final_state = None):
        if not self.is_master(): return
        if is_list(h_epi):
            ep_num = len(h_epi)
            ret = [None] * ep_num
            for i in range(ep_num):
                r = reward[i] if is_list(reward) else reward
                fs = final_state[i] if is_list(final_state) else final_state
                ret[i] = self.episode_feedback(h_epi[i], r, fs)
            return ret
        self.runtime_lock.lock('run')
        temp = self.episode_temp[h_epi]
        ret_h_epi = self.exp_rep.record(h_epi, temp['state'], temp['action'], reward, final_state)
        if ret_h_epi != h_epi:
            self.episode_temp[ret_h_epi] =  self.episode_temp.pop(h_epi)
        if final_state is not None:
            self.episode_temp.pop(ret_h_epi)
            ret_h_epi = None
        self.runtime_lock.unlock('run')
        return ret_h_epi

    def model_export(self, filename = None, h_mod = None):
        if not self.is_master():
            return
        if h_mod is None: h_mod = self._all_models()
        multi_mod = is_list(h_mod)
        if not multi_mod: h_mod = [h_mod]
        exported = [None] * len(h_mod)
        export_as_dict = (filename is None)
        for i in range(len(h_mod)):
            mod = self.master.get_model_monitor(h_mod[i])
            if mod is not None: exported[i] = request_export(mod)
            else: exported[i] = None
            f_name = filename[i] if multi_mod and not export_as_dict else filename
            if f_name is not None:
                path = get_file_path_from_name(f_name)
                try:
                    file = open(f_name, 'wb')
                except FileNotFoundError:
                    try:
                        makedirs(path, 0o777)
                        file = open(f_name, 'wb')
                    except:
                        file = None
                except:
                    file = None
                if file is not None:
                    dump_pack(exported[i], file)
                    file.close()
                else:
                    exported[i] = None
        return (exported if multi_mod else exported[0])

    def model_import(self, source, h_mod = None):
        if not self.is_master(): return
        if h_mod is None: h_mod = self._all_models()
        multi_mod = is_list(h_mod)
        if not multi_mod: h_mod = [h_mod]
        import_ret = [None] * len(h_mod)
        for i in range(len(h_mod)):
            src = source[i] if is_list(source) else source
            if isinstance(src, str):
                try: file = open(src, 'rb')
                except: file = None
                if file is not None:
                    src = load_pack(file)
                    file.close()
            if isinstance(src, dict):
                mod = self.master.get_model_monitor(h_mod[i])
                if mod is not None: import_ret[i] = request_import(mod, src)
                else: import_ret[i] = None
            else:
                import_ret[i] = None
        return (import_ret if multi_mod else import_ret[0])

    def runtime_clear(self):
        if not self.is_master(): return
        self.runtime_lock.lock('clear')
        self.exp_rep.clear()
        self.episode_temp.clear()
        self.runtime_lock.unlock('clear')

    def runtime_save(self, filename:str):
        if not self.is_master(): return
        self.runtime_lock.lock('save')
        pack = []
        path = get_file_path_from_name(filename)
        try: file = open(filename, 'wb')
        except FileNotFoundError:
            try:
                makedirs(path, 0o777)
                file = open(filename, 'wb')
            except: pack = None
        except: pack = None
        if pack is not None:
            pack = COPY([self.episode_temp])
            pack = pack + list(self.exp_rep.export_data())
            dump_pack(pack, file)
            file.close()
        self.runtime_lock.unlock('save')
        return pack

    def runtime_load(self, filename:str):
        if not self.is_master(): return
        self.runtime_lock.lock('load')
        pack = []
        try: file = open(filename, 'rb')
        except: pack = None
        if pack is not None:
            pack = load_pack(file)
            file.close()
            self.episode_temp.clear()
            self.episode_temp = pack[0]
            self.exp_rep.import_data(pack[1 : ])
        self.runtime_lock.unlock('load')
        return pack