__all__ = ['build']


from .__import__ import Runner
from .__import__ import Master
from .__import__ import align_notifier_session
from ..Notifiers.BatchPipe_M_S import build as build_MS_notifier
from ..Notifiers.BatchTunnel import build as build_tunnel
from ..Threads.BatchPipeM import build as build_thread


def build(master:Master) -> Runner:
    master.batch_pipe = batch_pipe = Runner()
    batch_pipe.notifiers = dict()
    for serv_id in master.server_monitors.keys():
        tunnel_key = -serv_id - 1
        notifier = batch_pipe.notifiers[serv_id] = build_MS_notifier(master, serv_id)
        tunnel = batch_pipe.notifiers[tunnel_key] = build_tunnel(master, serv_id)
        align_notifier_session(notifier, tunnel)
    batch_pipe.network = None
    batch_pipe.thread = build_thread(master)
    return batch_pipe