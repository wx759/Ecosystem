_all_ = ['new_notifier_context', 'del_notifier_context', 'register_descriptor', 'learn_descriptor_from_other'
         'Notifier', 'build_notifier', 'close_notifier', 'wait_notifier', 'notify', 'align_notifier_session']


from .Funcs import obj_to_bytes
from .Funcs import bytes_to_obj

from time import perf_counter
from time import time
from time import sleep
from threading import Semaphore
from threading import Thread
from functools import reduce
from .Locker import GroupLock

import socket

_PORT_MIN = 10000
_PORT_MAX = 65000

_PROC_PERIOD = 0.002
_CLIENT_QUEUE_LEN = 10
_TIME_OUT = 2
_T0 = time()
_PC0 = perf_counter()


_local_name = socket.gethostname()
_local_addr = socket.gethostbyname(_local_name)


_context_collection = {'mutex': Semaphore(1)}


Notifier = tuple


def _get_stamp():
    return perf_counter() - _PC0 + _T0


def _sleep_to(t:float):
    idle_time = t - _get_stamp()
    if idle_time > 0: sleep(idle_time)


def _process_wrapper(buffer:bytes, process_kernel, kernel_extra_args:list = None, kernel_ret:list = None):
    if kernel_extra_args is None: kernel_extra_args = list()
    while len(buffer) > 8:
        msg_len = int.from_bytes(buffer[ : 8], 'little')
        msg_ex_len = msg_len + 8
        if len(buffer) < msg_ex_len: break
        msg = buffer[8 : msg_ex_len]
        ret = process_kernel(msg, *kernel_extra_args)
        if isinstance(kernel_ret, list): kernel_ret.append(ret)
        buffer = buffer[msg_ex_len : ]
    return buffer


def _recv_from_socket(sock:socket.socket):
    data = b''
    try:
        sock.setblocking(False)
        recv_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    except: return data
    while True:
        try: rd = sock.recv(recv_buffer_size)
        except: rd = None
        if rd: data += rd
        else: break
    return data


def _send_to_socket(sock:socket.socket, data:bytes):
    try: sock.setblocking(False)
    except: return False
    t0 = _get_stamp()
    while len(data) > 0 and _get_stamp() - t0 < _TIME_OUT:
        try: sent_len = sock.send(data)
        except: sent_len = 0
        data = data[sent_len:]
    if len(data) > 0: return False
    else: return True


def _send_echos(sock:socket.socket, recv_stamp_bytes:list):
    data = b''
    for stamp_bytes in recv_stamp_bytes:
        len_bytes = len(stamp_bytes).to_bytes(8, 'little')
        data += len_bytes + stamp_bytes
    return _send_to_socket(sock, data)


def _acquire_connect(handler_context:dict, addr:str, port:int):
    sock = None
    while not handler_context['should_stop']:
        t0 = _get_stamp()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if sock.connect_ex((addr, port)) == 0: break
        sock.close()
        sock = None
        _sleep_to(t0 + _PROC_PERIOD)
    return sock


def _channel_proto():
    return {
        'mutex': Semaphore(1),
        'access': Semaphore(0),
        'stamp': 0,
        'note_bytes': None,
    }


def _write_note_bytes(channel, note_bytes, stamp):
    channel['mutex'].acquire()
    if stamp < 0 or channel['stamp'] < stamp:
        channel['stamp'] = stamp
        channel['note_bytes'] = note_bytes
        channel['access'].release()
    channel['mutex'].release()


def _read_note_bytes(channel):
    channel['access'].acquire()
    channel['mutex'].acquire()
    note_bytes = channel['note_bytes']
    while channel['access'].acquire(blocking = False): pass
    channel['mutex'].release()
    return note_bytes


def _process_kernel_msg(msg:bytes, channel_pool:dict):
    split_pos = msg.find(b'::')
    stamp_bytes = msg[: split_pos]
    stamp = float(stamp_bytes.decode())
    msg = msg[split_pos + 2:]
    split_pos = msg.find(b'::')
    key = msg[: split_pos].decode()
    note_bytes = msg[split_pos + 2:]
    try: channel = channel_pool[key]
    except KeyError:
        channel_pool['mutex'].acquire()
        try: channel = channel_pool[key]
        except KeyError: channel = channel_pool[key] = _channel_proto()
        channel_pool['mutex'].release()
    _write_note_bytes(channel, note_bytes, stamp)
    return stamp_bytes


def _process_kernel_echo(echo:bytes, msg_dict:dict):
    stamp = float(echo.decode())
    ret = True
    msg_dict['mutex'].acquire()
    try: msg_dict['sent'].pop(stamp)
    except KeyError: ret = False
    msg_dict['mutex'].release()
    return ret


def _check_time_out(msg_dict:dict):
    current_stamp = _get_stamp()
    ret = False
    msg_dict['mutex'].acquire()
    for sent in msg_dict['sent'].values():
        sent_stamp = sent[0]
        if current_stamp > sent_stamp + _TIME_OUT:
            ret = True
            break
    msg_dict['mutex'].release()
    return ret


def _enqueue_all_sent(msg_dict:dict):
    msg_dict['mutex'].acquire()
    for stamp, sent in msg_dict['sent'].items():
        msg_ex = sent[1]
        msg_dict['new'][stamp] = msg_ex
    msg_dict['sent'].clear()
    msg_dict['mutex'].release()


def _pack_new_msg(msg_dict:dict):
    out_buffer = b''
    sent_stamp = _get_stamp()
    msg_dict['mutex'].acquire()
    for stamp, msg_ex in msg_dict['new'].items():
        out_buffer += msg_ex
        msg_dict['sent'][stamp] = (sent_stamp, msg_ex)
    msg_dict['new'].clear()
    msg_dict['mutex'].release()
    return out_buffer


def _thread_func_income(handler_context: dict, channel_pool: dict, sock: socket.socket):
    in_buffer = b''
    while not handler_context['should_stop']:
        t0 = _get_stamp()
        in_buffer += _recv_from_socket(sock)
        recv_stamp_bytes = list()
        in_buffer = _process_wrapper(in_buffer, _process_kernel_msg,
                                     kernel_extra_args = [channel_pool], kernel_ret = recv_stamp_bytes)
        if not _send_echos(sock, recv_stamp_bytes): break
        _sleep_to(t0 + _PROC_PERIOD)
    sock.close()


def _thread_func_listener(listener:dict, port:int, income_handlers:dict, channel_pool:dict):
    while not listener['should_stop']:
        try: sock_listener = listener['socket']
        except KeyError:
            sock_listener = listener['socket'] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            sock_listener.bind((_local_addr, port))
            sock_listener.listen(_CLIENT_QUEUE_LEN)
        try: sock, addr = sock_listener.accept()
        except:
            try: listener.pop('socket').close()
            except: pass
            continue
        addr = addr[0]
        income_handlers['mutex'].acquire()
        if addr in income_handlers.keys():
            handler_context = income_handlers.pop(addr)
            handler_context['should_stop'] = True
            handler_context['thread'].join()
        handler_context = income_handlers[addr] = { 'should_stop': False }
        handler_context['thread'] = Thread(target =_thread_func_income, daemon = True,
                                           args = [handler_context, channel_pool, sock])
        handler_context['thread'].start()
        income_handlers['mutex'].release()
    if sock_listener is not None: sock_listener.close()


def _thread_func_outgo_sender(handler_context:dict):
    addr = handler_context['addr']
    port = handler_context['port']
    msg_dict = handler_context['msg_dict']
    sock_host = handler_context['sock_host']
    while not handler_context['should_stop']:
        t0 = _get_stamp()
        sock_host['mutex'].lock('reset')
        try: sock = sock_host['socket']
        except KeyError:
            sock_host['in_buffer'] = b''
            _enqueue_all_sent(msg_dict)
            sock = _acquire_connect(handler_context, addr, port)
            if sock is not None: sock_host['socket'] = sock
        sock_host['mutex'].unlock('reset')
        if sock is not None:
            data = _pack_new_msg(msg_dict)
            sock_host['mutex'].lock('run')
            if not _send_to_socket(sock, data):
                try: sock_host.pop('socket').close()
                except: pass
            sock_host['mutex'].unlock('run')
        _sleep_to(t0 + _PROC_PERIOD)
    sock_host['mutex'].lock('run')
    try: sock_host.pop('socket').close()
    except: pass
    sock_host['mutex'].unlock('run')


def _thread_func_outgo_echo(handler_context:dict):
    msg_dict = handler_context['msg_dict']
    sock_host = handler_context['sock_host']
    while not handler_context['should_stop']:
        t0 = _get_stamp()
        sock_host['mutex'].lock('run')
        try: sock = sock_host['socket']
        except KeyError: sock = None
        if sock is not None:
            sock_host['in_buffer'] += _recv_from_socket(sock)
            echo_checked = list()
            sock_host['in_buffer'] = _process_wrapper(sock_host['in_buffer'], _process_kernel_echo, [msg_dict], echo_checked)
            echo_checked = True if len(echo_checked) == 0 else reduce(bool.__and__, echo_checked)
            if not echo_checked:
                try: sock_host.pop('socket').close()
                except: pass
        sock_host['mutex'].unlock('run')
        _sleep_to(t0 + _PROC_PERIOD)


def _thread_func_outgo_timeout(handler_context:dict):
    msg_dict = handler_context['msg_dict']
    sock_host = handler_context['sock_host']
    while not handler_context['should_stop']:
        t0 = _get_stamp()
        sock_host['mutex'].lock('run')
        if 'socket' in sock_host.keys() and _check_time_out(msg_dict):
            try: sock_host.pop('socket').close()
            except: pass
        sock_host['mutex'].unlock('run')
        _sleep_to(t0 + _TIME_OUT / 2)


def _enqueue_msg(outgo_handlers:dict, addr:str, port:int, msg:bytes):
    stamp = _get_stamp()
    msg_ex = ('%.17f' % stamp).encode() + b'::' + msg
    msg_len_bytes = len(msg_ex).to_bytes(8, 'little')
    msg_ex = msg_len_bytes + msg_ex
    outgo_handlers['mutex'].acquire()
    if addr in outgo_handlers.keys():
        handler_context = outgo_handlers[addr]
    else:
        handler_context = outgo_handlers[addr] = {
            'msg_dict': {'mutex': Semaphore(1), 'new': dict(), 'sent': dict()},
            'sock_host': {'mutex': GroupLock(['run', 'reset']), 'in_buffer': b''},
            'addr': addr,
            'port': port,
            'should_stop': False
        }
        handler_context['sender'] = Thread(target = _thread_func_outgo_sender, args = [handler_context], daemon = True)
        handler_context['echo'] = Thread(target = _thread_func_outgo_echo, args = [handler_context], daemon = True)
        handler_context['timeout'] = Thread(target = _thread_func_outgo_timeout, args = [handler_context], daemon = True)
        handler_context['sender'].start()
        handler_context['echo'].start()
        handler_context['timeout'].start()
    outgo_handlers['mutex'].release()
    handler_context['msg_dict']['mutex'].acquire()
    handler_context['msg_dict']['new'][stamp] = msg_ex
    handler_context['msg_dict']['mutex'].release()
    return True


def _start_listener(context:dict):
    if context['listener']['thread'] is None:
        context['listener']['thread'] = Thread(target = _thread_func_listener, daemon = True,
                                          args = [context['listener'], context['port'], context['income_handlers'], context['channel_pool']])
        context['listener']['thread'].start()


def _port_in_range(port:int):
    return _PORT_MIN + (port - _PORT_MIN) % (_PORT_MAX - _PORT_MIN)


def _new_context(recomm_port:int, ret:list, idx:int, restrict:bool = True):
    t0 = _get_stamp()
    h_context = None
    port = _port_in_range(recomm_port)
    while h_context is None and _get_stamp() - t0 < _TIME_OUT:
        t1 = _get_stamp()
        _context_collection['mutex'].acquire()
        while not restrict and port in _context_collection.keys() and _get_stamp() - t0 < _TIME_OUT:
            port += 1
            port = _port_in_range(port)
        if port not in _context_collection.keys():
            h_context = port
            _context_collection[h_context] = {
                'port': port,
                'channel_pool': {'mutex': Semaphore(1)},
                'notifier_map': {'mutex': Semaphore(1)},
                'descriptor_map': {'mutex': Semaphore(1)},
                'income_handlers': {'mutex': Semaphore(1)},
                'outgo_handlers': {'mutex': Semaphore(1)},
                'listener': {'should_stop': False, 'thread': None}
            }
        _context_collection['mutex'].release()
        if h_context is None: _sleep_to(t1 + _PROC_PERIOD)
    ret[idx] = h_context


def _del_context(h_context:int, ret:list, idx:int):
    _context_collection['mutex'].acquire()
    try: context = _context_collection.pop(h_context)
    except KeyError: context = None
    _context_collection['mutex'].release()
    if context is not None:
        context['listener']['should_stop'] = True
        try: context['listener'].pop('socket').close()
        except: pass
        if context['listener']['thread'] is not None:
            context['listener']['thread'].join()
            context['listener']['thread'] = None
        for k, handler_context in context['income_handlers'].items():
            if k == 'mutex': continue
            handler_context['should_stop'] = True
            handler_context['thread'].join()
        for k, handler_context in context['outgo_handlers'].items():
            if k == 'mutex': continue
            handler_context['should_stop'] = True
            handler_context['timeout'].join()
            handler_context['echo'].join()
            handler_context['sender'].join()
        ret[idx] = True
    else:
        ret[idx] = None


def new_notifier_context(recomm_port, restrict:bool = True):
    if isinstance(recomm_port, int):
        ret = [None]
        _new_context(recomm_port, ret, 0, restrict)
        return ret[0]
    if isinstance(recomm_port, list) or isinstance(recomm_port, tuple):
        ret = [None] * len(recomm_port)
        threads = list()
        for idx in range(len(recomm_port)):
            threads.append(Thread(target = _new_context, args = [recomm_port[idx], ret, idx, restrict], daemon = True))
        for t in threads: t.start()
        for t in threads: t.join()
        if None in ret:
            del_notifier_context(ret)
            return None
        else:
            return ret
    return None


def del_notifier_context(h_context):
    if isinstance(h_context, int):
        ret = [None]
        _del_context(h_context, ret, 0)
        return ret[0]
    if isinstance(h_context, list) or isinstance(h_context, tuple):
        ret = [None] * len(h_context)
        threads = list()
        for idx in range(len(h_context)):
            threads.append(Thread(target = _del_context, args = [h_context[idx], ret, idx], daemon = True))
        for t in threads: t.start()
        for t in threads: t.join()
        return ret
    return None


def register_descriptor(h_context:int, descriptor:str, addr:str):
    try: context = _context_collection[h_context]
    except KeyError: return None
    descriptor_map = context['descriptor_map']
    descriptor_map['mutex'].acquire()
    descriptor_map[descriptor] = addr
    descriptor_map['mutex'].release()


def learn_descriptor_from_other(h_context:int, descriptor:str, other:int):
    try:
        context = _context_collection[h_context]
        context_0 = _context_collection[other]
    except KeyError: return None
    descriptor_map = context['descriptor_map']
    descriptor_map_0 = context_0['descriptor_map']
    descriptor_map['mutex'].acquire()
    descriptor_map_0['mutex'].acquire()
    try:
        descriptor_map[descriptor] = descriptor_map_0[descriptor]
    except KeyError: pass
    descriptor_map_0['mutex'].release()
    descriptor_map['mutex'].release()


def align_notifier_session(notifier:Notifier, align_to:Notifier):
    h_context, name = notifier
    h_context0, name0 = align_to
    try:
        context0 = _context_collection[h_context0]
        context = _context_collection[h_context]
        notifier_entity0 = context0['notifier_map'][name0]
        notifier_entity = context['notifier_map'][name]
    except KeyError: return None
    notifier_entity0['mutex'].acquire()
    notifier_entity['mutex'].acquire()
    notifier_entity['session_stamp'] = notifier_entity0['session_stamp']
    notifier_entity['mutex'].release()
    notifier_entity0['mutex'].release()


def build_notifier(h_context:int, scope:str, near_name:str, far_name:str,
                   far_descriptor:str, with_new_session:bool = False) -> Notifier:
    try: context = _context_collection[h_context]
    except KeyError: return None
    name = near_far_key = ('%s||%s->%s' % (scope, near_name, far_name)).replace('::', '..')
    far_near_key = ('%s||%s->%s' % (scope, far_name, near_name)).replace('::', '..')
    far_addr = context['descriptor_map'][far_descriptor]
    if far_addr == _local_addr:
        context['channel_pool']['mutex'].acquire()
        if near_far_key not in context['channel_pool'].keys():
            context['channel_pool'][near_far_key] = _channel_proto()
        context['channel_pool']['mutex'].release()
        notify_channel_key = near_far_key
        remote = False
    else:
        notify_channel_key = far_addr
        remote = True
        _start_listener(context)
    context['channel_pool']['mutex'].acquire()
    if far_near_key not in context['channel_pool'].keys():
        context['channel_pool'][far_near_key] = _channel_proto()
    context['channel_pool']['mutex'].release()
    wait_channel_key = far_near_key
    session_stamp = _get_stamp() if with_new_session else -1
    context['notifier_map']['mutex'].acquire()
    context['notifier_map'][name] = {
        'wait_channel_key': wait_channel_key,
        'notify_channel_key': notify_channel_key,
        'remote': remote,
        'session_stamp': session_stamp,
        'mutex': Semaphore(1)
    }
    context['notifier_map']['mutex'].release()
    return (h_context, name)


def notify(notifier:Notifier, note = True):
    h_context, name = notifier
    try:
        context = _context_collection[h_context]
        notifier_entity = context['notifier_map'][name]
    except KeyError: return None
    notify_channel_key = notifier_entity['notify_channel_key']
    note_bytes = obj_to_bytes((notifier_entity['session_stamp'], note))
    if notifier_entity['remote']:
        addr = notify_channel_key
        msg = name.encode() + b'::' + note_bytes
        ret = _enqueue_msg(context['outgo_handlers'], addr, context['port'], msg)
    else:
        try: notify_channel = context['channel_pool'][notify_channel_key]
        except KeyError:
            context['channel_pool']['mutex'].acquire()
            try: notify_channel = context['channel_pool'][notify_channel_key]
            except KeyError: notify_channel = context['channel_pool'][notify_channel_key] = _channel_proto()
            context['channel_pool']['mutex'].release()
        _write_note_bytes(notify_channel, note_bytes, _get_stamp())
        ret = True
    return ret


def wait_notifier(notifier:Notifier):
    h_context, name = notifier
    try:
        context = _context_collection[h_context]
        notifier_entity = context['notifier_map'][name]
        wait_channel_key = notifier_entity['wait_channel_key']
        wait_channel = context['channel_pool'][wait_channel_key]
    except KeyError: return None
    acquired = False
    while not acquired:
        rd = bytes_to_obj(_read_note_bytes(wait_channel))
        if rd is None: return None
        s_stamp, note = rd
        if s_stamp < 0: break
        notifier_entity['mutex'].acquire()
        if s_stamp >= notifier_entity['session_stamp']:
            notifier_entity['session_stamp'] = s_stamp
            acquired = True
        notifier_entity['mutex'].release()
    return note


def close_notifier(notifier:Notifier):
    h_context, name = notifier
    try: context = _context_collection[h_context]
    except KeyError: return None
    ret = True
    context['notifier_map']['mutex'].acquire()
    context['channel_pool']['mutex'].acquire()
    try:
        notifier_entity = context['notifier_map'].pop(name)
        wait_channel = context['channel_pool'].pop(notifier_entity['wait_channel_key'])
    except KeyError: ret = None
    context['channel_pool']['mutex'].release()
    context['notifier_map']['mutex'].release()
    if ret: _write_note_bytes(wait_channel, obj_to_bytes(None), -1)
    return ret