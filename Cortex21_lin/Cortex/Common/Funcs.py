__all__ = ['get_file_path_from_name', 'is_list', 'hash_str', 'make_tensorbord_runner',
           'obj_to_str', 'str_to_obj', 'obj_to_bytes', 'bytes_to_obj']


from os import makedirs
from platform import system
from hashlib import md5
from _pickle import dumps
from _pickle import loads


def get_file_path_from_name(filename:str):
    filename = filename.replace('\\', '/')
    last_slash_at = filename.rfind('/')
    if last_slash_at == 0: return '/'
    if last_slash_at < 0: return ''
    else: return filename[0 : last_slash_at]


def is_list(d):
    return isinstance(d, list)


def hash_str(s:str):
    m = md5()
    m.update(s.encode())
    return int(m.hexdigest(), 16)


def make_tensorbord_runner(path:str):
    filename = 'tensorboard_here'
    if 'Windows' in system():
        filename = filename + '.bat'
    else:
        filename = filename + '.sh'
    try: file = open(path + '/' + filename, 'w')
    except FileNotFoundError:
        try:
            makedirs(path, 0o777)
            file = open(path + '/' + filename, 'w')
        except: return False
    except: return False
    file.writelines('tensorboard --logdir=.')
    file.close()
    return True


def obj_to_str(obj) -> str:
    b = dumps(obj)
    return b.hex()


def str_to_obj(s):
    if isinstance(s, bytes):
        s = s.decode()
    b = bytes.fromhex(s)
    return loads(b)


def obj_to_bytes(obj) -> bytes:
    b = dumps(obj)
    return b


def bytes_to_obj(b):
    obj = loads(b)
    return obj

