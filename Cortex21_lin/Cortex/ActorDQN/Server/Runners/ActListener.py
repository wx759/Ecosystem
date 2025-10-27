__all__ = ['build']


from .__import__ import Runner
from .__import__ import Model
from ..Notifiers.ActRequest import build as build_act_req_notifier
from ..Networks.ActListener import build as build_net
from ..Threads.ActListener import build as build_thread


def build(mod:Model) -> Runner:
    mod.act_listener = listener = Runner()
    serv = mod.host.host
    master_embed = serv.is_master()
    listener.notifiers = dict() if master_embed else {'request': build_act_req_notifier(mod)}
    listener.network = build_net(mod)
    listener.thread = None if master_embed else build_thread(mod)
    return listener