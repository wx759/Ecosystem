__all__ = ['build']


from .__import__ import Runner
from .__import__ import Device
from ..Notifiers.DispatchRequest import build as build_dispatch_req_notifier
from ..Threads.DispatchListener import build as build_thread


def build(dev:Device) -> Runner:
    dev.dispatch_listener = listener = Runner()
    serv = dev.host
    master_embed = serv.is_master()
    listener.notifiers = dict() if master_embed else {'request': build_dispatch_req_notifier(dev)}
    listener.thread = None if master_embed else build_thread(dev)
    return listener