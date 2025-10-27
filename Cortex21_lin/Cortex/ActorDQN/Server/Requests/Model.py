__all__ = ['request_act', 'request_train', 'request_reset', 'request_import', 'request_export']


from .__import__ import wait_notifier
from .__import__ import notify
from .__import__ import ModelMonitor
from .__import__ import IDLE_TIME_SHORT
from .__import__ import FLAG_NO_BATCH
from ..Networks.ActListener import do_act
from ..Networks.TrainListener import do_train
from ..Networks.StorageListener import do_export
from ..Networks.StorageListener import do_import
from ..Networks.StorageListener import do_reset

from time import sleep
from numpy import array


def request_act(mod_mnt:ModelMonitor, state:array, initial_state:list):
    while not mod_mnt.settled: sleep(IDLE_TIME_SHORT)
    master = mod_mnt.master
    mod_mnt.request_lockers['act'].lock()
    if mod_mnt.host is not None:
        embed_mod = mod_mnt.embed_obj
        if embed_mod is None:
            input = {'request':'act', 'state': state, 'intial_state': initial_state}
            act_notifier = mod_mnt.act_notifier
            notify(act_notifier, input)
            ret = wait_notifier(act_notifier)
        else:
            ret = do_act(embed_mod.act_listener.network, state, initial_state)
    else:
        ret = None
    mod_mnt.request_lockers['act'].unlock()
    return ret


def request_train(mod_mnt:ModelMonitor, learning_rate:float, dropout_rate_dict:dict):
    while not mod_mnt.settled: sleep(IDLE_TIME_SHORT)
    master = mod_mnt.master
    if not master.batch_history.has_record(): return FLAG_NO_BATCH
    mod_mnt.request_lockers['train'].lock()
    if mod_mnt.host is not None:
        embed_mod = mod_mnt.embed_obj
        if embed_mod is None:
            input = {'request': 'train', 'learning_rate': learning_rate, 'dropout_rate_dict': dropout_rate_dict}
            train_notifier = mod_mnt.train_notifier
            notify(train_notifier, input)
            ret = wait_notifier(train_notifier)
        else:
            ret = do_train(embed_mod.train_listener.network, learning_rate, dropout_rate_dict)
    else:
        ret = None
    mod_mnt.request_lockers['train'].unlock()
    return ret


def request_export(mod_mnt:ModelMonitor):
    while not mod_mnt.settled: sleep(IDLE_TIME_SHORT)
    master = mod_mnt.master
    mod_mnt.request_lockers['storage'].lock()
    if mod_mnt.host is not None:
        embed_mod = mod_mnt.embed_obj
        if embed_mod is None:
            input = {'request': 'export'}
            storage_notifier = mod_mnt.storage_notifier
            notify(storage_notifier, input)
            ret = wait_notifier(storage_notifier)
        else:
            ret = do_export(embed_mod.storage_listener.network)
    else:
        ret = None
    mod_mnt.request_lockers['storage'].unlock()
    return ret


def request_import(mod_mnt:ModelMonitor, src:dict):
    while not mod_mnt.settled: sleep(IDLE_TIME_SHORT)
    master = mod_mnt.master
    mod_mnt.request_lockers['storage'].lock()
    if mod_mnt.host is not None:
        embed_mod = mod_mnt.embed_obj
        if embed_mod is None:
            input = {'request': 'import', 'src': src}
            storage_notifier = mod_mnt.storage_notifier
            notify(storage_notifier, input)
            ret = wait_notifier(storage_notifier)
        else:
            ret = do_import(embed_mod.storage_listener.network, src)
    else:
        ret = None
    mod_mnt.request_lockers['storage'].unlock()
    return ret


def request_reset(mod_mnt:ModelMonitor):
    while not mod_mnt.settled: sleep(IDLE_TIME_SHORT)
    master = mod_mnt.master
    mod_mnt.request_lockers['storage'].lock()
    if mod_mnt.host is not None:
        embed_mod = mod_mnt.embed_obj
        if embed_mod is None:
            input = {'request': 'reset'}
            storage_notifier = mod_mnt.storage_notifier
            notify(storage_notifier, input)
            ret = wait_notifier(storage_notifier)
        else:
            ret = do_reset(embed_mod.storage_listener.network)
    else:
        ret = None
    mod_mnt.request_lockers['storage'].unlock()
    return ret