__all__ = ['build', 'close']


from .__import__ import Model
from .__import__ import Device
from .__import__ import close_notifier
from .Runner import close as close_runner
from ..Runners.TrainListener import build as build_train_listener
from ..Runners.ActListener import build as build_act_listener
from ..Runners.StorageListener import build as build_storage_listener
from ..Runners.BatchPipeL import build as build_batch_pipe


def build(id:int, dev:Device) -> Model:
    mod = Model(id)
    mod.host = dev
    mod.train_listener = build_train_listener(mod)
    dev.batch_pipe.mutex.lock()
    mod.batch_pipe = build_batch_pipe(mod)
    dev.batch_pipe.mutex.unlock()
    mod.act_listener = build_act_listener(mod)
    mod.storage_listener = build_storage_listener(mod)
    return mod


def close(mod:Model):
    dev = mod.host
    mod_id = mod.id
    try: close_notifier(dev.batch_pipe.notifiers[mod_id])
    except KeyError: pass
    if mod.batch_pipe is not None:
        close_runner(mod.batch_pipe)
    if mod.act_listener is not None:
        close_runner(mod.act_listener)
    if mod.train_listener is not None:
        close_runner(mod.train_listener)
    if mod.storage_listener is not None:
        close_runner(mod.storage_listener)
    dev.batch_pipe.mutex.lock()
    dev.batch_pipe.notifiers.pop(mod_id)
    dev.batch_pipe.mutex.unlock()
    mod.act_listener = None
    mod.train_listener = None
    mod.storage_listener = None
    mod.batch_pipe = None

