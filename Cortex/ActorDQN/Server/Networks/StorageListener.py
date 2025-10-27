__all__ = ['build', 'do_export', 'do_import', 'do_reset']


from .__import__ import Network
from .__import__ import VarFilter
from .__import__ import Model
from .__import__ import IDLE_TIME_SHORT
from .__import__ import FLAG_FULL_IMPORTED
from .__import__ import FLAG_PARTIAL_IMPORTED
from .Scopes import *

from time import sleep


def build(mod:Model) -> dict:
    network = {
        'host': mod,
        'BatchPipe': mod.batch_pipe.network,
        'Trainer': mod.train_listener.network
    }
    return network


def do_export(network:dict):
    mod:Model = network['host']
    while not mod.initialized: sleep(IDLE_TIME_SHORT)
    mod.locker.lock()
    BP_net:Network = network['BatchPipe']
    TR_net:Network = network['Trainer']
    exported = BP_net.session_export_variables()
    exported_trainer = TR_net.session_export_variables(var_filters = VarFilter.for_trainer_only)
    for k, v in exported_trainer.items(): exported[k] = v
    mod.locker.unlock()
    if len(exported) > 0:
        return exported
    else:
        return None


def do_import(network:dict, src:dict):
    mod:Model = network['host']
    mod.locker.lock()
    BP_net:Network = network['BatchPipe']
    TR_net:Network = network['Trainer']
    imported = BP_net.session_import_variables(src)
    imported_trainer = TR_net.session_import_variables(src, var_filters = VarFilter.for_trainer_only)
    for k, v in imported_trainer.items(): imported[k] = v
    if len(imported) > 0:
        full_imported_num = len(BP_net._var_dict)
        for k, v in TR_net._var_dict.items():
            if v['for_trainer']: full_imported_num += 1
        trained_step_name = BP_net.key_tensors['trained_step'].name
        if trained_step_name in imported.keys():
            trained_step = imported[trained_step_name]
            config = mod.host.host.config
            if trained_step % config.upd_shadow_period == 0:
                BP_net.shadow_need_update = True
        else:
            full_imported_num -= 1
        if len(imported) == full_imported_num:
            ret = FLAG_FULL_IMPORTED
            mod.initialized = True
        else:
            ret = FLAG_PARTIAL_IMPORTED
        BP_net.q_target_expired = True
    else:
        ret = None
    mod.locker.unlock()
    return ret


def do_reset(network:dict):
    mod:Model = network['host']
    mod.locker.lock()
    BP_net = network['BatchPipe']
    TR_net = network['Trainer']
    ret = BP_net.session_reset_global() and TR_net.session_reset_trainer()
    if ret:
        mod.initialized = True
        BP_net.shadow_need_update = False
        BP_net.q_target_expired = True
    else:
        ret = None
    mod.locker.unlock()
    return ret


def r_net_only(name:str, var_info:dict = None):
    return name.find(GLOBAL_SCOPE_R_NET) >= 0


def not_r_net(name:str, var_info:dict = None):
    return not r_net_only(name, var_info)


VarFilter.r_net_only = r_net_only
VarFilter.not_r_net = not_r_net
