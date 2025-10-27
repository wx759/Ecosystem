__all__ = ['build', 'do_close', 'do_dispatch', 'do_withdraw']


from .__import__ import Thread
from .__import__ import Device
from .__import__ import wait_notifier
from .__import__ import notify
from ..Networks.StorageListener import do_export
from ..Networks.StorageListener import do_import
from ..Networks.StorageListener import do_reset
from ..Factory.Model import build as build_model
from ..Factory.Model import close as close_model


def build(dev:Device) -> Thread:
    t = Thread(target = thread_func, args = [dev], daemon = True)
    t.start()
    return t


def thread_func(dev:Device):
    dispatch_listener = dev.dispatch_listener
    while not dispatch_listener.should_stop:
        request = wait_notifier(dispatch_listener.notifiers['request'])
        if not isinstance(request, dict): continue
        if request['request'] == 'dispatch':
            mod_id = request['mod_id']
            src = request['src']
            ret = do_dispatch(dev, mod_id, src)
        if request['request'] == 'withdraw':
            mod_id = request['mod_id']
            ret = do_withdraw(dev, mod_id)
        notify(dispatch_listener.notifiers['request'], ret)


def do_dispatch(dev:Device, mod_id:int, src:dict):
    mod = build_model(mod_id, dev)
    if src is None:
        ret = do_reset(mod.storage_listener.network)
    else:
        ret = do_import(mod.storage_listener.network, src)
    dev.models[mod_id] = mod
    dev.host.on_server_mod_num += 1
    dev.host.print_log('Dispatch: Model #%d ---> Device #%d' % (mod_id, dev.id), level = 1)
    return ret


def do_withdraw(dev:Device, mod_id:int):
    mod = dev.models.pop(mod_id)
    dev.host.on_server_mod_num -= 1
    ret = do_export(mod.storage_listener.network)
    close_model(mod)
    dev.host.print_log('Withdraw: Model #%d // Device #%d' % (mod_id, dev.id), level = 1)
    return ret
