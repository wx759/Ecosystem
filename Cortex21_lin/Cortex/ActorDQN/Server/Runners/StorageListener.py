__all__ = ['build']


from .__import__ import Runner
from .__import__ import Model
from ..Notifiers.StorageRequest import build as build_storage_req_notifier
from ..Networks.StorageListener import build as build_net
from ..Threads.StorageListener import build as build_thread


def build(mod:Model) -> Runner:
    mod.storage_listener = listener = Runner()
    serv = mod.host.host
    master_embed = serv.is_master()
    listener.notifiers = dict() if master_embed else {'request': build_storage_req_notifier(mod)}
    listener.network = build_net(mod)
    listener.thread = None if master_embed else build_thread(mod)
    return listener