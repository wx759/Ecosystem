__all__ = ['build']


from .__import__ import Runner
from .__import__ import Server
from ..Notifiers.BatchPipe_M_S import build as build_MS_notifier
from ..Notifiers.BatchPipe_S_D import build as build_SD_notifier
from ..Networks.BatchPipeS import build as build_net
from ..Threads.BatchPipeS import build as build_thread


def build(serv:Server, dev_ids) -> Runner:
    serv.batch_pipe = batch_pipe = Runner()
    batch_pipe.notifiers = {
        dev_id: build_SD_notifier(serv, dev_id) for dev_id in dev_ids
    }
    batch_pipe.notifiers['upper'] = build_MS_notifier(serv)
    batch_pipe.tunnel_not_built = True
    batch_pipe.network = build_net(serv)
    batch_pipe.thread = build_thread(serv)
    return batch_pipe