__all__ = ['build']


from .__import__ import Device
from .__import__ import Server
from .__import__ import Notifier
from .__import__ import build_notifier
from .__import__ import server_descriptor
from .Scopes import NOTIFIER_SCOPE


def build(near, dev_id = None) -> Notifier:
    if isinstance(near, Server):
        serv = near
        near_name = 'S%d' % serv.id
        far_name = 'D%d' % dev_id
        near_server = serv
        with_new_session = True
    if isinstance(near, Device):
        dev = near
        near_name = 'D%d' % dev.id
        near_server = dev.host
        far_name = 'S%d' % near_server.id
        with_new_session = False
    scope = NOTIFIER_SCOPE('BatchPipe/S_D')
    far_descriptor = server_descriptor(near_server)
    return build_notifier(near_server.h_notifier_context, scope, near_name, far_name, far_descriptor, with_new_session)