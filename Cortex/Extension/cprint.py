__all__ = ['cprint']


import platform
from ctypes import POINTER, WINFUNCTYPE, windll
from ctypes.wintypes import DWORD, BOOL, HANDLE


color_name = ['black', 'red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white']
color_code = range(len(color_name))
color = {n : c for n, c in zip(color_name, color_code)}

if 'Windows' in platform.system():
    _std_out_handle = -11
    _virtual_terminal_processing_enabled = 0x0004

    GetStdHandle = WINFUNCTYPE(HANDLE, DWORD)(('GetStdHandle', windll.kernel32))
    GetConsoleMode = WINFUNCTYPE(BOOL, HANDLE, POINTER(DWORD))(('GetConsoleMode', windll.kernel32))
    SetConsoleMode = WINFUNCTYPE(BOOL, HANDLE, DWORD)(('SetConsoleMode', windll.kernel32))
    handle = GetStdHandle(_std_out_handle)
    mode = DWORD()
    if GetConsoleMode(handle, mode):
        mode = DWORD(mode.value | _virtual_terminal_processing_enabled)
        SetConsoleMode(handle, mode)

def cprint(line:str, front_color = None, back_color = None, bold = False, underline = False, end = '\n'):
    codes = []
    if front_color in color_name:
        codes.append('%d' % (30 + color[front_color]))
    if back_color in color_name:
        codes.append('%d' % (40 + color[back_color]))
    if bold:
        codes.append('1')
    if underline:
        codes.append('4')
    code = ';'.join(codes)
    prefix = ''
    postfix = ''
    if len(codes) > 0:
        prefix = '\033[%sm' % code
        postfix = '\033[0m'
    print('%s%s%s' % (prefix, line, postfix), end = end)

