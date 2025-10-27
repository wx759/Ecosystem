__all__ = ['EventTimeRecorder', 'SectionTimer', 'new_section_timer', 'get_section_timer', 'pop_section_timer']


from time import perf_counter
from .Locker import BasicLock


class EventTimeRecorder:
    def __init__(self):
        self.locker = BasicLock()
        self.time = 0
        self.event_num = 0

    def record_event_time(self, t):
        self.locker.lock()
        self.time += t
        self.event_num += 1
        self.locker.unlock()

    def reset(self):
        self.locker.lock()
        self.time = 0
        self.event_num = 0
        self.locker.unlock()

    def get_time_per_event(self) -> float:
        self.locker.lock()
        tpe = self.time / self.event_num if self.event_num > 0 else -1
        self.locker.unlock()
        return tpe


class SectionTimer:
    def __init__(self):
        self.time = 0
        self.locker = BasicLock()
        self.in_section_num = 0
        self.t0 = 0

    def enter_timing_section(self):
        self.locker.lock()
        if self.in_section_num == 0:
            self.t0 = perf_counter()
        self.in_section_num += 1
        self.locker.unlock()

    def exit_timing_section(self):
        self.locker.lock()
        assert self.in_section_num > 0
        self.in_section_num -= 1
        if self.in_section_num == 0:
            self.time += perf_counter() - self.t0
        self.locker.unlock()

    def reset(self):
        self.locker.lock()
        assert self.in_section_num == 0
        self.time = 0
        self.locker.unlock()

    def get_time(self):
        return self.time


__timer_pool = dict()
__ticket = 0
__mutex = BasicLock()


def new_section_timer():
    __mutex.lock()
    global __ticket
    this_ticket = __ticket
    __ticket += 1
    timer = __timer_pool[this_ticket] = SectionTimer()
    __mutex.unlock()
    return (this_ticket, timer)

def get_section_timer(ticket):
    __mutex.lock()
    try: timer = __timer_pool[ticket]
    except KeyError: timer = None
    __mutex.unlock()
    return timer

def pop_section_timer(ticket):
    __mutex.lock()
    try: timer = __timer_pool.pop(ticket)
    except KeyError: timer = None
    __mutex.unlock()
    return timer
