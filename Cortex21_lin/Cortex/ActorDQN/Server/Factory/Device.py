__all__ = ['build', 'close']


from .__import__ import Device
from .__import__ import Server
from ..Runners.DispatchListener import build as build_dispatch_listener
from ..Runners.BatchPipeD import build as build_batch_pipe
from .Runner import close as close_runner
from .Model import close as close_model


def build(id:int, tf_code:str, serv:Server) -> Device:
    dev = Device(id, tf_code, serv)
    dev.batch_pipe = build_batch_pipe(dev)
    dev.dispatch_listener = build_dispatch_listener(dev)
    return dev


def close(dev:Device):
    for mod in dev.models.values(): close_model(mod)
    if dev.dispatch_listener is not None:
        close_runner(dev.dispatch_listener)
    if dev.batch_pipe is not None:
        close_runner(dev.batch_pipe)
    dev.models.clear()
    dev.dispatch_listener = None
    dev.batch_pipe = None

