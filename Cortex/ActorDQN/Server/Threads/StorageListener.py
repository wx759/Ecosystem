__all__ = ['build']


from .__import__ import Thread
from .__import__ import Model
from .__import__ import wait_notifier
from .__import__ import notify
from ..Networks.StorageListener import do_export
from ..Networks.StorageListener import do_import
from ..Networks.StorageListener import do_reset


def build(mod:Model) -> Thread:
    t = Thread(target = thread_func, args = [mod], daemon = True)
    t.start()
    return t


def thread_func(mod:Model):
    storage_listener = mod.storage_listener
    while not storage_listener.should_stop:
        request = wait_notifier(storage_listener.notifiers['request'])
        if not isinstance(request, dict): continue
        if request['request'] == 'export':
            ret = do_export(storage_listener.network)
        if request['request'] == 'import':
            src = request['src']
            ret = do_import(storage_listener.network, src)
        if request['request'] == 'reset':
            ret = do_reset(storage_listener.network)
        notify(storage_listener.notifiers['request'], ret)
