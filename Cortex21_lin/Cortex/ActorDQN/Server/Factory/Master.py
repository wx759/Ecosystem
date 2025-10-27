__all__ = ['build', 'close']


from .__import__ import Master
from .__import__ import Server
from .__import__ import ModelMonitor
from .__import__ import ServerMonitor
from .__import__ import read_cluster_config
from .__import__ import Experience_Replay as ExpRep
from .__import__ import new_notifier_context
from .__import__ import NET_PORT_ERROR
from .__import__ import JOB_NAME
from .DeviceMonitor import build as build_device_monitor
from .DeviceMonitor import close as close_device_monitor
from .ModelMonitor import close as close_model_monitor
from .Runner import close as close_runner
from ..Runners.BatchPipeM import build as build_batch_pipe
from ..Threads.Dispatcher import build as build_dispatcher


def build(serv:Server, exp_rep:ExpRep, h_ps:list = None, mod_num:int = 1, cluster_config:dict = None) -> Master:
    serv.h_tunnel_context = new_notifier_context(serv.h_notifier_context, restrict = False)
    if serv.h_tunnel_context is None: raise NET_PORT_ERROR
    master = Master(serv, exp_rep, h_ps)
    _, device_map, master.dispatch_interval, _ = read_cluster_config(cluster_config)
    if device_map is None:
        master.server_monitors[0] = ServerMonitor(0, master)
        for dev in serv.devices.values():
            master.device_monitors[dev.id] = build_device_monitor(dev.id, master.server_monitors[0])
    else:
        dev_id_start = 0
        for serv_id, d_num in zip(range(len(device_map)), device_map.values()):
            if d_num == 0: d_num = 1
            master.server_monitors[serv_id] = ServerMonitor(serv_id, master)
            for dev_id in range(dev_id_start, dev_id_start + d_num):
                master.device_monitors[dev_id] = build_device_monitor(dev_id, master.server_monitors[serv_id])
            dev_id_start += d_num
    master.model_monitors = {mod_id: ModelMonitor(mod_id, master) for mod_id in range(mod_num)}
    master.batch_pipe = build_batch_pipe(master)
    master.dispatcher = build_dispatcher(master)
    return master


def close(master:Master):
    master.should_stop = True
    if master.dispatcher is not None:
        master.dispatcher.join()
    if master.batch_pipe is not None:
        close_runner(master.batch_pipe)
    for mod in master.model_monitors.values():
        close_model_monitor(mod)
    for dev in master.device_monitors.values():
        close_device_monitor(dev)
    master.dispatcher = None
    master.batch_pipe = None
    master.model_monitors.clear()
    master.device_monitors.clear()
    master.server_monitors.clear()
