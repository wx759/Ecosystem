__all__ = ['get_file_path_from_name', 'is_list', 'hash_str', 'make_tensorbord_runner',
           'obj_to_str', 'str_to_obj', 'obj_to_bytes', 'bytes_to_obj','float2int16','int_list2float','action_encode','action_decode']


from os import makedirs
from platform import system
from hashlib import md5
from _pickle import dumps
from _pickle import loads
import numpy as np


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

def float2fix(f, int_bit=3, decimal_bit=12):
    # print(f)
    bit_num = int_bit + decimal_bit + 1
    prec = 1 / 2 ** decimal_bit  # 精度
    # 显示范围
    # decimal_max = (2 ** 12 - 1) * prec
    # num_max = 2 ** int_bit - 1 + decimal_max
    # num_min = - 2 ** int_bit - 1 + decimal_max
    # print("The value range of fixed point:({0},{1})".format(num_min,num_max))
    if f > 0:
        sign = "0"
    else:
        sign = "1"

    f = abs(f)
    data = int(f // prec)
    quotient = ""  # 余数
    while True:
        remainder = data // 2
        quotients = data % 2
        quotient = quotient + str(quotients)
        if remainder == 0:
            break
        else:
            data = remainder
    add_bit = bit_num - 1 - len(quotient)
    if add_bit != 0:
        quotient = quotient + "0" * add_bit + sign
    else:
        quotient = quotient + sign
    str_bin = quotient[::-1]
    return str_bin


# 将十进制浮点转化为二进制定点原码S12
def fix2float(str_bin, int_bit=3, decimal_bit=12):
    decimal = 0

    string_int = str_bin[1:int_bit + 1]
    string_decimal = str_bin[int_bit + 1:]
    for i in range(len(string_decimal)):
        decimal += 2 ** (-i - 1) * int(string_decimal[i])
    data = int(str(int(string_int, 2))) + decimal
    if str_bin[0] == '1':

        return -data
    else:
        return data

def float2int16(f):
    return int(float2fix(f),2)

def int_list2float(int_list,isFour):
    b_a = [[] for i in range(len(int_list))]
    for pos in range(len(int_list)):
        b_a[pos].append(fix2float(format(int_list[pos][0], "b").zfill(16)))
    return np.array(b_a)

def action_encode(action):
    a = 0
    for i in range(len(action[0])):
        action[0][i] = int(action[0][i] * 100) / 100
        a *= 100
        a += int((action[0][i] + 0.5) * 100)
    # a = round(action[0][0] + 0.5, 1) / 100 + round(action[0][1] + 0.5, 1) / 10 + round(action[0][2] + 0.5,1) + round(action[0][3] + 0.5, 1) * 10
    return a

def action_decode(action):
    temp = []
    for i in range(4):
        temp.insert(0,(action % 100) / 100 - 0.5)
        action //= 100
    return temp
