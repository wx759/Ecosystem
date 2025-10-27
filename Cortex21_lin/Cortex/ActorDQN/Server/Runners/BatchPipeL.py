__all__ = ['build']


from .__import__ import Runner
from .__import__ import Model
from .__import__ import notify
from ..Notifiers.BatchPipe_D_L import build as build_DL_notifier
from ..Notifiers.BatchPipe_L_T import build as build_LT_notifier
from ..Networks.BatchPipeL import build as build_net
from ..Threads.BatchPipeL import build as build_thread


def build(mod:Model) -> Runner:
    mod.batch_pipe = batch_pipe = Runner()
    mod_id = mod.id
    dev = mod.host
    dev.batch_pipe.notifiers[mod_id] = build_DL_notifier(dev, mod_id)
    batch_pipe.notifiers = {
        'upper': build_DL_notifier(mod),
        'lower': build_LT_notifier(mod, True)
    }
    notify(batch_pipe.notifiers['upper'], -1)
    batch_pipe.network = build_net(mod)
    batch_pipe.thread = build_thread(mod)
    return batch_pipe