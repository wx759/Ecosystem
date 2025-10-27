__all__ = ['build']


from .__import__ import Thread
from .__import__ import Model
from .__import__ import wait_notifier
from .__import__ import notify
from ..Networks.ActListener import do_act


def build(mod:Model) -> Thread:
    t = Thread(target = thread_func, args = [mod], daemon = True)
    t.start()
    return t


def thread_func(mod:Model):
    act_listener = mod.act_listener
    while not act_listener.should_stop:
        request = wait_notifier(act_listener.notifiers['request'])
        if not isinstance(request, dict): continue
        if request['request'] == 'act':
            state = request['state']
            initial_state = request['intial_state']
            ret = do_act(act_listener.network, state, initial_state)
        notify(act_listener.notifiers['request'], ret)


