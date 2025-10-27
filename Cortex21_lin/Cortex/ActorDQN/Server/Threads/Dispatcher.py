__all__ = ['build']


from .__import__ import Thread
from .__import__ import Master
from .__import__ import EventTimeRecorder
from .__import__ import DeviceMonitor
from .__import__ import ModelMonitor
from .__import__ import IDLE_TIME_LONG
from ..Factory.ModelMonitor import settle as settle_model_monitor
from ..Factory.ModelMonitor import close as close_model_monitor
from ..Requests.Device import request_dispatch_model
from ..Requests.Device import request_withdraw_model

from time import time
from time import sleep
from numpy import argmax
from numpy import argmin
from random import sample


def build(master:Master) -> Thread:
    dispatch_all(master)
    t = Thread(target = thread_func, args = [master], daemon = True)
    t.start()
    return t


def thread_func(master:Master):
    t0 = time()
    timing = False
    while not master.should_stop:
        sleep(IDLE_TIME_LONG)
        if (not timing) and (time() - t0 > master.dispatch_interval * 0.2):
            for dev_mnt in master.device_monitors.values():
                dev_mnt.time_recorder.reset()
            timing = True
        if time() - t0 > master.dispatch_interval:
            rearrange(master)
            timing = False
            t0 = time()
    withdraw_all(master)


def dispatch_all(master:Master):
    dev_num = len(master.device_monitors)
    threads = list()
    for mod_mnt in master.model_monitors.values():
        dev_dst = master.device_monitors[mod_mnt.id % dev_num]
        t = Thread(target = move_model, args = [mod_mnt, dev_dst], daemon = True)
        threads.append(t)
        t.start()
    for t in threads: t.join()


def withdraw_all(master:Master):
    threads = list()
    for mod_mnt in master.model_monitors.values():
        t = Thread(target = move_model, args = [mod_mnt, None], daemon = True)
        threads.append(t)
        t.start()
    for t in threads: t.join()


def rearrange(master:Master):
    avg_time = [-1] * len(master.device_monitors)
    for dev_id in master.device_monitors.keys():
        time_recorder:EventTimeRecorder = master.device_monitors[dev_id].time_recorder
        avg_time[dev_id] = time_recorder.get_time_per_event()
    i_max = argmax(avg_time)
    max_t = max(avg_time)
    i_min = argmin([t if t >= 0 else max_t + 1 for t in avg_time])
    if i_min == i_max: return
    if avg_time[i_max] > 1.01 * avg_time[i_min]:
        dev_src = master.device_monitors[i_max]
        dev_dst = master.device_monitors[i_min]
        mod_mnts = list(dev_src.model_monitors.values())
        mod_mnt = sample(mod_mnts, 1)[0]
        move_model(mod_mnt, dev_dst)


def move_model(mod_mnt:ModelMonitor, dev_dst:DeviceMonitor):
    for locker in mod_mnt.request_lockers.values(): locker.lock()
    mod_id = mod_mnt.id
    dev_src = mod_mnt.host
    if dev_src is not None:
        exported = request_withdraw_model(dev_src, mod_id)
        assert (exported is not None)
        close_model_monitor(mod_mnt)
        dev_src.model_monitors.pop(mod_id)
    else: exported = None
    if dev_dst is not None:
        dispatch_ret = request_dispatch_model(dev_dst, mod_id, exported)
        assert (dispatch_ret is not None)
        dev_dst.model_monitors[mod_id] = mod_mnt
        settle_model_monitor(mod_mnt, dev_dst)
    for locker in mod_mnt.request_lockers.values(): locker.unlock()
