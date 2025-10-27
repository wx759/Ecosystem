__all__ = ['build', 'close']


from .__import__ import DeviceMonitor
from .__import__ import ServerMonitor
from .__import__ import close_notifier
from .ModelMonitor import close as close_model_monitor
from ..Notifiers.DispatchRequest import build as build_dispatch_notifier


def build(id:int, serv_mnt:ServerMonitor) -> DeviceMonitor:
    dev = DeviceMonitor(id, serv_mnt)
    dev.dispatch_notifier = build_dispatch_notifier(dev)
    return dev


def close(dev:DeviceMonitor):
    if dev.dispatch_notifier is not None:
        close_notifier(dev.dispatch_notifier)
    for mod in dev.model_monitors.values(): close_model_monitor(mod)
    dev.dispatch_notifier = None
    dev.model_monitors.clear()
