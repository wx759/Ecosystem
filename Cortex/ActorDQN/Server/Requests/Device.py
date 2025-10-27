__all__ = ['request_dispatch_model', 'request_withdraw_model']


from .__import__ import wait_notifier
from .__import__ import notify
from .__import__ import DeviceMonitor
from ..Threads.DispatchListener import do_dispatch
from ..Threads.DispatchListener import do_withdraw


def request_dispatch_model(dev_mnt:DeviceMonitor, mod_id:int, src:dict):
    dev_mnt.request_locker.lock()
    embed_dev = dev_mnt.embed_obj
    if embed_dev is None:
        input = {'request': 'dispatch', 'mod_id': mod_id, 'src': src}
        dispatch_notifier = dev_mnt.dispatch_notifier
        notify(dispatch_notifier, input)
        ret = wait_notifier(dispatch_notifier)
    else:
        ret = do_dispatch(embed_dev, mod_id, src)
    dev_mnt.request_locker.unlock()
    return ret


def request_withdraw_model(dev_mnt:DeviceMonitor, mod_id:int):
    dev_mnt.request_locker.lock()
    embed_dev = dev_mnt.embed_obj
    if embed_dev is None:
        input = {'request': 'withdraw', 'mod_id': mod_id}
        dispatch_notifier = dev_mnt.dispatch_notifier
        notify(dispatch_notifier, input)
        ret = wait_notifier(dispatch_notifier)
    else:
        ret = do_withdraw(embed_dev, mod_id)
    dev_mnt.request_locker.unlock()
    return ret