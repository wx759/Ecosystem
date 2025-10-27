__all__ = ['build', 'close']


from socket import gethostname
from socket import gethostbyname
from multiprocessing import cpu_count

from .__import__ import tf
from .__import__ import hash_str
from .__import__ import Server
from .__import__ import Config
from .__import__ import Network
from .__import__ import Thread
from .__import__ import new_notifier_context
from .__import__ import del_notifier_context
from .__import__ import register_descriptor
from .__import__ import read_cluster_config
from .__import__ import JOB_NAME
from .__import__ import NET_PORT_ERROR
from .__import__ import TF_THREADS_PER_CPU
from .Device import build as build_device
from .Device import close as close_device
from .Runner import close as close_runner
from ..Runners.BatchPipeS import build as build_batch_pipe


def _register_all_descriptors(h_notifier_context:int, device_map:dict, group_name:str):
    for serv_id, addr in zip(range(len(device_map)), device_map.keys()):
        descriptor = JOB_NAME(group_name, serv_id)
        register_descriptor(h_notifier_context, descriptor, addr)


def _make_tf_server(serv:Server, tf_inter_op_threads:int):
    tf_server_port = new_notifier_context(serv.h_notifier_context, restrict = False)
    if tf_server_port is None: raise NET_PORT_ERROR
    job_name = JOB_NAME(serv.group_name, serv.id)
    tf_cluster = {job_name: ['localhost:%d' % tf_server_port]}
    tf_cluster_spec = tf.train.ClusterSpec(tf_cluster)
    tf_config = tf.ConfigProto(inter_op_parallelism_threads = tf_inter_op_threads)
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True
    serv.server = tf.train.Server(tf_cluster_spec, job_name, 0, config = tf_config)


def build(mod_num:int = 1, config:Config = None, cluster_config:dict = None):
    serv = Server(config)
    local_name = gethostname()
    serv.address = gethostbyname(local_name)
    dev_num = 0
    dev_id_start = 0
    serv.group_name, device_map, _, serv.log_level = read_cluster_config(cluster_config)
    if device_map is None:
        dev_num = len(Network.get_gpu_info())
        dev_id_start = 0
        serv.id = 0
        device_map = {serv.address: dev_num}
    else:
        device_map = {(serv.address if (k == 'localhost' or k == local_name) else k): v
                       for k, v in device_map.items()}
        for i, item in zip(range(len(device_map)), device_map.items()):
            addr, d_num = item
            if serv.address == addr:
                serv.id = i
                dev_num = d_num
                break
            else:
                if d_num == 0: d_num = 1
                dev_id_start += d_num
    if serv.id >= 0:
        serv.h_notifier_context = new_notifier_context(hash_str(serv.group_name), restrict = True)
        if serv.h_notifier_context is None: raise NET_PORT_ERROR
        _register_all_descriptors(serv.h_notifier_context, device_map, serv.group_name)
        if dev_num == 0:
            dev_type = 'cpu'
            d_num = 1
        else:
            dev_type = 'gpu'
            d_num = dev_num
        serv.tf_inter_op_threads = cpu_count() * TF_THREADS_PER_CPU
        _make_tf_server(serv, serv.tf_inter_op_threads)
        dev_ids = range(dev_id_start, dev_id_start + d_num)
        serv.batch_pipe = build_batch_pipe(serv, dev_ids)
        serv.devices = {d: build_device(d, '/%s:%d' % (dev_type, d - dev_id_start), serv) for d in dev_ids}
    return serv


def close(serv:Server):
    threads = []
    for dev in serv.devices.values():
        t = Thread(target = close_device, args = [dev], daemon = True)
        t.start()
        threads.append(t)
    if serv.batch_pipe is not None:
        t = Thread(target = close_runner, args = [serv.batch_pipe], daemon = True)
        t.start()
        threads.append(t)
    for t in threads: t.join()
    serv.devices.clear()
    serv.batch_pipe = None
    if serv.h_notifier_context is not None:
        del_notifier_context(serv.h_notifier_context)
        serv.h_notifier_context = None
    if serv.h_tunnel_context is not None:
        del_notifier_context(serv.h_tunnel_context)
        serv.h_tunnel_context = None
    serv.stoping_locker.unlock()
