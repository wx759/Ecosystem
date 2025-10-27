__all__ = ['build']


from .__import__ import Runner
from .__import__ import Model
from ..Notifiers.TrainRequest import build as build_train_req_notifier
from ..Notifiers.BatchPipe_L_T import build as build_LT_notifier
from ..Networks.TrainListener import build as build_net
from ..Threads.TrainListener import build as build_thread


def build(mod:Model) -> Runner:
    mod.train_listener = listener = Runner()
    listener.notifiers = { 'batch_pipe': build_LT_notifier(mod, False)}
    listener.network = build_net(mod)
    serv = mod.host.host
    if serv.is_master():
        listener.thread = None
    else:
        listener.notifiers['request'] = build_train_req_notifier(mod)
        listener.thread = build_thread(mod)
    return listener