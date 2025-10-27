__all__ = ['close']


from .__import__ import Runner
from .__import__ import close_notifier
from .__import__ import Network


def close(runner:Runner):
    runner.should_stop = True
    for notifier in runner.notifiers.values():
        close_notifier(notifier)
    if isinstance(runner.network, Network):
        runner.network.session_close()
    if (runner.thread is not None) and runner.thread.is_alive():
        runner.thread.join()
    runner.notifiers.clear()
    runner.network = None
    runner.thread = None