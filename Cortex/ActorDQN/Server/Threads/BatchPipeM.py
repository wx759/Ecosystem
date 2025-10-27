__all__ = ['build']


from .__import__ import Thread
from .__import__ import Master
from .__import__ import wait_notifier
from .__import__ import notify
from .__import__ import IDLE_TIME_SHORT

from numpy import concatenate
from random import shuffle
from time import sleep


def build(master:Master) -> Thread:
    t = Thread(target = thread_func, args = [master], daemon = True)
    t.start()
    return t


def thread_func(master:Master):
    batch_pipe = master.batch_pipe
    serv = master.serv
    config = serv.config
    first_batch = True
    while not batch_pipe.should_stop:
        batch = get_batch(master)
        if batch is None:
            sleep(IDLE_TIME_SHORT)
            continue
        pick_epi, pick_pos = batch[-2 : ]
        batch_ticket = master.record_batch_info(pick_epi, pick_pos)
        batch = batch[ : -2]
        batch.append(batch_ticket)
        serv_ids = master.server_monitors.keys()
        if first_batch:
            first_batch = False
        else:
            for serv_id in serv_ids:
                T = wait_notifier(batch_pipe.notifiers[serv_id])
                if T is not None:
                    for dev_id, t in T.items():
                        master.device_monitors[dev_id].time_recorder.record_event_time(t)
        send_batch_to_tunnel(master, batch)
        for serv_id in serv_ids:
            notify(batch_pipe.notifiers[serv_id], serv.h_tunnel_context)


def append_batch(batch, append):
    if append is None: return
    for i in range(len(batch)):
        if batch[i] is None:
            batch[i] = append[i]
        else:
            batch[i] = concatenate((batch[i], append[i]))


def get_batch(master:Master):
    config = master.serv.config
    batch_size = config.train_batch_size
    valid_sample_rate = config.valid_sample_rate
    if master.h_ps is not None:
        mod_num = len(master.model_monitors)
        mod_ids = list(range(mod_num))
        shuffle(mod_ids)
        batch_size_mod = batch_size // mod_num
        valid_sample_rate *= mod_num
        batch = [None, None, None, None, None, None, None, None]
        for i in range(mod_num):
            mod_id = mod_ids[i]
            h_ps = master.h_ps[mod_id]
            if i == mod_num - 1: batch_size_mod += batch_size % mod_num
            batch_mod = master.exp_rep.get_random_batch(batch_size_mod, valid_sample_rate, h_ps)
            append_batch(batch, batch_mod)
        if batch[0] is None: batch = None
    else:
        batch = master.exp_rep.get_random_batch(batch_size, valid_sample_rate)
    return batch


def send_batch_to_tunnel(master:Master, batch):
    for serv_id in master.server_monitors.keys():
        tunnel_key = -serv_id - 1
        notify(master.batch_pipe.notifiers[tunnel_key], batch)