__all__ = ['build']


from .__import__ import Thread
from .__import__ import Device
from .__import__ import wait_notifier
from .__import__ import notify
from .__import__ import SectionTimer
from .__import__ import get_section_timer
from ..Networks.BatchPipeD import do_flow

from numpy import average


def build(dev:Device) -> Thread:
    t = Thread(target = thread_func, args = [dev], daemon = True)
    t.start()
    return t


def thread_func(dev:Device):
    batch_pipe = dev.batch_pipe
    while not batch_pipe.should_stop:
        t_timer_fw = wait_notifier(batch_pipe.notifiers['upper'])
        if t_timer_fw is not None:
            batch_pipe.mutex.lock()
            notifiers = [notifier for key, notifier in batch_pipe.notifiers.items() if not (key == 'upper')]
            batch_pipe.mutex.unlock()
            t_timer_rep = None
            for notifier in notifiers:
                t_timer = wait_notifier(notifier)
                assert (t_timer_rep is None) or (t_timer is None) or (t_timer < 0) or (t_timer == t_timer_rep)
                if t_timer is not None and t_timer >= 0: t_timer_rep = t_timer
            timer_fw:SectionTimer = get_section_timer(t_timer_fw)
            do_flow(batch_pipe.network, timer_fw)
            for notifier in notifiers: notify(notifier, t_timer_fw)
            notify(batch_pipe.notifiers['upper'], t_timer_rep)

