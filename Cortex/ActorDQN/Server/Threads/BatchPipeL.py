__all__ = ['build']


from .__import__ import Thread
from .__import__ import Model
from .__import__ import wait_notifier
from .__import__ import notify
from .__import__ import SectionTimer
from .__import__ import get_section_timer
from ..Networks.BatchPipeL import do_flow


def build(mod:Model) -> Thread:
    t = Thread(target = thread_func, args = [mod], daemon = True)
    t.start()
    return t


def thread_func(mod:Model):
    batch_pipe = mod.batch_pipe
    first_batch = True
    while not batch_pipe.should_stop:
        t_timer_fw = wait_notifier(batch_pipe.notifiers['upper'])
        if t_timer_fw is not None:
            if first_batch:
                first_batch = False
                t_timer_rep = -1
            else:
                t_timer_rep = wait_notifier(batch_pipe.notifiers['lower'])
            timer_fw:SectionTimer = get_section_timer(t_timer_fw)
            flag = do_flow(batch_pipe.network, timer_fw)
            notify(batch_pipe.notifiers['lower'], (flag, t_timer_fw))
            notify(batch_pipe.notifiers['upper'], t_timer_rep)

