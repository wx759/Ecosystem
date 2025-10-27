__all__ = ['BasicLock', 'GroupLock']


from threading import Semaphore


def _lock(mutex:Semaphore, blocking:bool = True):
    return mutex.acquire(blocking = blocking)

def _unlock(mutex:Semaphore):
    try:
        mutex.release()
    except RuntimeError: return False
    return True


class BasicLock():
    def __init__(self, init_locked:bool = False):
        init_val = 0 if init_locked else 1
        self._mutex = Semaphore(init_val)

    def lock(self, blocking:bool = True):
        return _lock(self._mutex, blocking)

    def unlock(self):
        return _unlock(self._mutex)


class GroupLock():
    def __init__(self, groups:list):
        assert len(groups) > 1
        self._counter = {g: {'num': 0, 'mutex': Semaphore(1)} for g in groups}
        self._mutex = Semaphore(1)

    def lock(self, group):
        _lock(self._counter[group]['mutex'])
        if self._counter[group]['num'] == 0: _lock(self._mutex)
        self._counter[group]['num'] += 1
        _unlock(self._counter[group]['mutex'])

    def unlock(self, group):
        _lock(self._counter[group]['mutex'])
        self._counter[group]['num'] -= 1
        if self._counter[group]['num'] == 0: _unlock(self._mutex)
        _unlock(self._counter[group]['mutex'])
