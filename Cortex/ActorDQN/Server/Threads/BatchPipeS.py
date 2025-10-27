__all__ = ['build']


from .__import__ import Thread
from .__import__ import Server
from .__import__ import wait_notifier
from .__import__ import notify
from .__import__ import SectionTimer
from .__import__ import new_section_timer
from .__import__ import pop_section_timer
from .__import__ import close_notifier
from .__import__ import del_notifier_context
from .__import__ import new_notifier_context
from .__import__ import align_notifier_session
from .__import__ import NET_PORT_ERROR
from ..Notifiers.BatchTunnel import build as build_tunnel
from ..Networks.BatchPipeS import do_flow


def build(serv:Server) -> Thread:
    t = Thread(target = thread_func, args = [serv], daemon = True)
    t.start()
    return t


def thread_func(serv:Server):
    batch_pipe = serv.batch_pipe
    first_batch = True
    while not batch_pipe.should_stop:
        tunnel_port = wait_notifier(batch_pipe.notifiers['upper'])
        if tunnel_port is not None:
            T = dict()
            if serv.on_server_mod_num > 0:
                dev_ids = serv.devices.keys()
                if first_batch:
                    first_batch = False
                else:
                    for dev_id in dev_ids:
                        t_timer_rep = wait_notifier(batch_pipe.notifiers[dev_id])
                        timer_rep:SectionTimer = pop_section_timer(t_timer_rep)
                        if timer_rep is not None: T[dev_id] = timer_rep.get_time()
                if tunnel_port != serv.h_tunnel_context:
                    rebuild_tunnel_context(serv, tunnel_port)
                if batch_pipe.tunnel_not_built:
                    batch_pipe.notifiers['tunnel'] = build_tunnel(serv)
                    batch_pipe.tunnel_not_built = False
                timer_fw_dict = dict()
                for dev_id in dev_ids:
                    timer_fw_dict[dev_id] = new_section_timer()
                    timer_fw_dict[dev_id][1].enter_timing_section()
                align_notifier_session(batch_pipe.notifiers['tunnel'], batch_pipe.notifiers['upper'])
                batch = wait_notifier(batch_pipe.notifiers['tunnel'])
                if batch is not None:
                    do_flow(batch_pipe.network, batch)
                for dev_id in dev_ids:
                    timer_fw_dict[dev_id][1].exit_timing_section()
                    notify(batch_pipe.notifiers[dev_id], timer_fw_dict[dev_id][0])
            notify(batch_pipe.notifiers['upper'], T)


def rebuild_tunnel_context(serv:Server, tunnel_port:list):
    if 'tunnel' in serv.batch_pipe.notifiers.keys():
        close_notifier(serv.batch_pipe.notifiers.pop('tunnel'))
    del_notifier_context(serv.h_tunnel_context)
    serv.h_tunnel_context = new_notifier_context(tunnel_port, restrict = True)
    if serv.h_tunnel_context is None: raise NET_PORT_ERROR
    serv.batch_pipe.tunnel_not_built = True
