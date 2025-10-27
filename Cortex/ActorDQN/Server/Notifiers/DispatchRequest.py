__all__ = ['build']


from .__import__ import Device
from .__import__ import DeviceMonitor
from .__import__ import Notifier
from .__import__ import build_notifier
from .__import__ import server_descriptor
from .__import__ import master_descriptor
from .Scopes import NOTIFIER_SCOPE


def build(near) -> Notifier:
    if isinstance(near, DeviceMonitor):
        dev_mnt = near
        if dev_mnt.embed_obj is not None: return None
        serv_mnt = dev_mnt.host
        near_name = 'master'
        far_name = 'D%d' % dev_mnt.id
        near_server = dev_mnt.master.serv
        far_descriptor = server_descriptor(serv_mnt)
        with_new_session = True
    if isinstance(near, Device):
        dev = near
        serv = dev.host
        near_name = 'D%d' % dev.id
        far_name = 'master'
        near_server = serv
        far_descriptor = master_descriptor(serv)
        with_new_session = False
    scope = NOTIFIER_SCOPE('Request/Dispatch')
    return build_notifier(near_server.h_notifier_context, scope, near_name, far_name, far_descriptor, with_new_session)
