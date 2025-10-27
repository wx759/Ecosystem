__all__ = ['Runner', 'Model', 'ModelMonitor', 'Device', 'DeviceMonitor', 'Server', 'ServerMonitor', 'Master', 'Thread',
           'Config', 'Network', 'ExpRep', 'read_cluster_config', 'JOB_NAME', 'server_descriptor',
           'master_descriptor', 'DISPATCH_INTERVAL_DEFAULT', 'DISPATCH_INTERVAL_MIN',
           'IDLE_TIME_SHORT', 'IDLE_TIME_LONG', 'NET_PORT_ERROR', 'TF_THREADS_PER_CPU']


from .__import__ import tf
from .__import__ import TF_Neural_Network as Network
from .__import__ import Notifier
from .__import__ import BasicLock
from .__import__ import Config
from .__import__ import Experience_Replay as ExpRep
from .__import__ import EventTimeRecorder

from time import time
from time import sleep
from copy import deepcopy
from threading import Thread


DISPATCH_INTERVAL_DEFAULT = 180
DISPATCH_INTERVAL_MIN = 30

IDLE_TIME_SHORT = 0.005
IDLE_TIME_LONG = 1

NET_PORT_ERROR = ValueError('Net port occupied.')

TF_THREADS_PER_CPU = 2


def JOB_NAME(group_name, serv_id):
    return '%s_%s' % (group_name, serv_id)


def read_cluster_config(cluster_config:dict):
    stamp = int(time()*10000) % 1000000
    if not isinstance(cluster_config, dict):
        cluster_config = {
            'group': 'local%d' % stamp,
            'devices': None,
            'dispatch_interval': DISPATCH_INTERVAL_DEFAULT,
            'log_level': 0
        }
    else: cluster_config = deepcopy(cluster_config)
    if 'group' not in cluster_config.keys():
        cluster_config['group'] = 'local%d' % stamp
        if 'devices' not in cluster_config:
            cluster_config['devices'] = None
        else:
            if len(cluster_config['devices']) > 1:
                raise ValueError('A group name is needed for the cluster.')
    if 'devices' not in cluster_config.keys() or not isinstance(cluster_config['devices'], dict) or len(cluster_config['devices']) == 0:
        cluster_config['devices'] = None
    if 'dispatch_interval' not in cluster_config.keys():
        cluster_config['dispatch_interval'] = DISPATCH_INTERVAL_DEFAULT
    if cluster_config['dispatch_interval'] < DISPATCH_INTERVAL_MIN:
        cluster_config['dispatch_interval'] = DISPATCH_INTERVAL_MIN
    if 'log_level' not in cluster_config.keys():
        cluster_config['log_level'] = 0
    return (cluster_config['group'], cluster_config['devices'], cluster_config['dispatch_interval'], cluster_config['log_level'])


def server_descriptor(serv_proxy):
    if isinstance(serv_proxy, Server):
        group_name = serv_proxy.group_name
    if isinstance(serv_proxy, ServerMonitor):
        group_name = serv_proxy.master.serv.group_name
    return JOB_NAME(group_name, serv_proxy.id)


def master_descriptor(serv_proxy):
    if isinstance(serv_proxy, Server):
        group_name = serv_proxy.group_name
    if isinstance(serv_proxy, ServerMonitor):
        group_name = serv_proxy.master.serv.group_name
    return JOB_NAME(group_name, 0)


def print_graph(network:Network, path:str):
    try:
        network.logs_path(path, clear = True, make_tensorbord_runner = False)
        network.print_graph()
    except: pass


class Runner:
    def __init__(self):
        self.network:Network = None
        self.thread:Thread = None
        self.notifiers = dict()
        self.should_stop = False


class BatchHistory:
    def __init__(self, size:int):
        self.size = size
        self.locker = BasicLock()
        self.pool = [None] * size
        self.ticket = -1

    def record(self, pick_epi, pick_pos) -> int:
        self.locker.lock()
        self.ticket = (self.ticket + 1) % self.size
        batch_ticket = self.ticket
        self.pool[batch_ticket] = (pick_epi, pick_pos)
        self.locker.unlock()
        return batch_ticket

    def get_batch_info(self, batch_ticket: int) -> tuple:
        self.locker.lock()
        batch_info = self.pool[batch_ticket]
        self.locker.unlock()
        return batch_info

    def has_record(self):
        return self.ticket >= 0


class Server:
    def __init__(self, config:Config = None):
        if config is None:
            self.config = Config()
        else:
            self.config = config
        self.group_name:str = None
        self.log_level:int = 0
        self.tf_inter_op_threads = 0
        self.on_server_mod_num = 0
        self.id:int = -1
        self.h_notifier_context:int = None
        self.h_tunnel_context:int = None
        self.devices = dict()
        self.stoping_locker = BasicLock(True)
        self.address:str = None
        self.server:tf.train.Server = None
        self.batch_pipe: Runner = None

    def stand_by(self):
        t = Thread(target = self.stoping_locker.lock, daemon = False)
        t.start()

    def is_master(self):
        return (self.id == 0)

    def print_log(self, log: str, level: int = 1):
        if level <= self.log_level:
            print(log)

    def print_graphs_threads(self, graph_root:str):
        threads = []
        t = Thread(target = print_graph, daemon = True,
                   args = [self.batch_pipe.network, graph_root + '/BatchPipe/Server%d/' % self.id])
        t.start()
        threads.append(t)
        devs = list(self.devices.values())
        for dev in devs:
            threads += dev.print_graphs_threads(graph_root)
        return threads


class Device:
    def __init__(self, id:int, tf_code:str, serv:Server):
        self.id = id
        self.host = serv
        self.tf_code = tf_code
        self.models = dict()
        self.dispatch_listener:Runner = None
        self.batch_pipe:Runner = None

    def print_graphs_threads(self, graph_root:str):
        threads = []
        t = Thread(target = print_graph, daemon = True,
                   args = [self.batch_pipe.network, graph_root + '/BatchPipe/Device%d/' % self.id])
        t.start()
        threads.append(t)
        mods = list(self.models.values())
        for mod in mods:
            threads += mod.print_graphs_threads(graph_root)
        return threads


class Model:
    def __init__(self, id:int):
        self.id = id
        self.host:Device = None
        self.initialized = False
        self.act_listener:Runner = None
        self.train_listener:Runner = None
        self.storage_listener:Runner = None
        self.batch_pipe:Runner = None
        self.locker = BasicLock()

    def print_graphs_threads(self, graph_root:str):
        threads = []
        t = Thread(target = print_graph, daemon = True,
                   args = [self.act_listener.network, graph_root + '/Actors/Model_%d/' % self.id])
        t.start()
        threads.append(t)
        t = Thread(target = print_graph, daemon = True,
                   args = [self.train_listener.network, graph_root + '/Trainers/Model_%d/' % self.id])
        t.start()
        threads.append(t)
        t = Thread(target = print_graph, daemon = True,
                   args = [self.batch_pipe.network, graph_root + '/BatchPipe/Model_%d/' % self.id])
        t.start()
        threads.append(t)
        return threads


class Master:
    def __init__(self, serv:Server, exp_rep:ExpRep, h_ps:list):
        self.serv = serv
        self.exp_rep:ExpRep = exp_rep
        self.h_ps = h_ps
        self.model_monitors = dict()
        self.device_monitors = dict()
        self.server_monitors = dict()
        self.dispatch_interval = DISPATCH_INTERVAL_DEFAULT
        self.batch_history = BatchHistory(20)
        self.should_stop = False
        self.batch_pipe:Runner = None
        self.dispatcher:Thread = None

    def record_batch_info(self, pick_epi, pick_pos) -> int:
        return self.batch_history.record(pick_epi, pick_pos)

    def get_batch_info(self, batch_ticket: int) -> tuple:
        return self.batch_history.get_batch_info(batch_ticket)

    def get_model_monitor(self, mod_id):
        try: mod_mnt = self.model_monitors[mod_id]
        except KeyError: mod_mnt = None
        return mod_mnt

    def model_num(self):
        return len(self.model_monitors)

    def wait_all_models_settled(self):
        mod_mnts = list(self.model_monitors.values())
        for mod_mnt in mod_mnts:
            while not mod_mnt.settled: sleep(IDLE_TIME_SHORT)


class ServerMonitor:
    def __init__(self, id:int, master:Master):
        self.id = id
        self.master = master

    def is_master(self):
        return (self.id == 0)


class DeviceMonitor:
    def __init__(self, id:int, serv_mnt:ServerMonitor):
        self.id = id
        self.master = serv_mnt.master
        self.host = serv_mnt
        self.embed_obj:Device = serv_mnt.master.serv.devices[id] if serv_mnt.is_master() else None
        self.request_locker = BasicLock()
        self.time_recorder = EventTimeRecorder()
        self.model_monitors = dict()
        self.dispatch_notifier:Notifier = None


class ModelMonitor:
    def __init__(self, id:int, master:Master):
        self.id = id
        self.master = master
        self.host:DeviceMonitor = None
        self.settled = False
        self.embed_obj:Model = None
        self.request_lockers = {'act': BasicLock(), 'train': BasicLock(), 'storage': BasicLock()}
        self.act_notifier:Notifier = None
        self.train_notifier:Notifier = None
        self.storage_notifier:Notifier = None
