__all__ = ['build']


from .__import__ import Model
from .__import__ import ModelMonitor
from .__import__ import Notifier
from .__import__ import build_notifier
from .__import__ import server_descriptor
from .__import__ import master_descriptor
from .Scopes import NOTIFIER_SCOPE


def build(near) -> Notifier:
    if isinstance(near, ModelMonitor):
        mod_mnt = near
        if mod_mnt.embed_obj is not None: return None
        dev_mnt = mod_mnt.host
        serv_mnt = dev_mnt.host
        near_name = 'master'
        far_name = 'M%dD%d' % (mod_mnt.id, dev_mnt.id)
        near_server = mod_mnt.master.serv
        far_descriptor = server_descriptor(serv_mnt)
        with_new_session = True
    if isinstance(near, Model):
        mod = near
        dev = mod.host
        serv = dev.host
        near_name = 'M%dD%d' % (mod.id, dev.id)
        far_name = 'master'
        near_server = serv
        far_descriptor = master_descriptor(serv)
        with_new_session = False
    scope = NOTIFIER_SCOPE('Request/Storage')
    return build_notifier(near_server.h_notifier_context, scope, near_name, far_name, far_descriptor, with_new_session)


