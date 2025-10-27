__all__ = ['build']


from .__import__ import Server
from .__import__ import Master
from .__import__ import Notifier
from .__import__ import build_notifier
from .__import__ import server_descriptor
from .__import__ import master_descriptor
from .Scopes import NOTIFIER_SCOPE


def build(near, serv_id = None) -> Notifier:
    if isinstance(near, Master):
        master = near
        serv_mnt = master.server_monitors[serv_id]
        near_name = 'master'
        far_name = 'S%d' % serv_id
        near_server = master.serv
        far_descriptor = server_descriptor(serv_mnt)
        with_new_session = True
    if isinstance(near, Server):
        serv = near
        near_name = 'S%d' % serv.id
        far_name = 'master'
        near_server = serv
        far_descriptor = master_descriptor(serv)
        with_new_session = False
    scope = NOTIFIER_SCOPE('BatchPipe/M_S')
    return build_notifier(near_server.h_notifier_context, scope, near_name, far_name, far_descriptor, with_new_session)
