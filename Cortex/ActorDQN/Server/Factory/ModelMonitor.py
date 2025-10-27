__all__ = ['settle', 'close']


from .__import__ import ModelMonitor
from .__import__ import DeviceMonitor
from .__import__ import close_notifier
from ..Notifiers.ActRequest import build as build_act_notifier
from ..Notifiers.TrainRequest import build as build_train_notifier
from ..Notifiers.StorageRequest import build as build_storage_notifier


def settle(mod_mnt:ModelMonitor, dev_mnt:DeviceMonitor):
    mod_mnt.host = dev_mnt
    if dev_mnt.embed_obj is None:
        mod_mnt.embed_obj = None
    else:
        mod_mnt.embed_obj = dev_mnt.embed_obj.models[mod_mnt.id]
    mod_mnt.act_notifier = build_act_notifier(mod_mnt)
    mod_mnt.train_notifier = build_train_notifier(mod_mnt)
    mod_mnt.storage_notifier = build_storage_notifier(mod_mnt)
    mod_mnt.settled = True


def close(mod_mnt:ModelMonitor):
    if mod_mnt.act_notifier is not None:
        close_notifier(mod_mnt.act_notifier)
    if mod_mnt.train_notifier is not None:
        close_notifier(mod_mnt.train_notifier)
    if mod_mnt.storage_notifier is not None:
        close_notifier(mod_mnt.storage_notifier)
    mod_mnt.host = None
    mod_mnt.embed_obj = None
    mod_mnt.train_notifier = None
    mod_mnt.act_notifier = None
    mod_mnt.storage_notifier = None
