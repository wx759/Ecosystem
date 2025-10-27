__all__ = ['build']


from .__import__ import Thread
from .__import__ import Model
from .__import__ import wait_notifier
from .__import__ import notify
from ..Networks.TrainListener import do_train


def build(mod:Model) -> Thread:
    t = Thread(target = thread_func, args = [mod], daemon = True)
    t.start()
    return t


def thread_func(mod:Model):
    train_listener = mod.train_listener
    while not train_listener.should_stop:
        request = wait_notifier(train_listener.notifiers['request'])
        if not isinstance(request, dict): continue
        if request['request'] == 'train':
            learning_rate = request['learning_rate']
            dropout_rate_dict = request['dropout_rate_dict']
            ret = do_train(train_listener.network, learning_rate, dropout_rate_dict)
        notify(train_listener.notifiers['request'], ret)

