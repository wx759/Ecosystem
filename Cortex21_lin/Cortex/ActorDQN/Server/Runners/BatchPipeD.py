__all__ = ['build']


from .__import__ import Runner
from .__import__ import Device
from .__import__ import BasicLock
from ..Notifiers.BatchPipe_S_D import build as build_SD_notifier
from ..Networks.BatchPipeD import build as build_net
from ..Threads.BatchPipeD import build as build_thread


def build(dev:Device) -> Runner:
    dev.batch_pipe = batch_pipe = Runner()
    batch_pipe.mutex = BasicLock()
    batch_pipe.notifiers = { 'upper': build_SD_notifier(dev) }
    batch_pipe.network = build_net(dev)
    batch_pipe.thread = build_thread(dev)
    return batch_pipe