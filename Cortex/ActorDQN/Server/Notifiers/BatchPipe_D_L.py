__all__ = ['build']


from .__import__ import Device
from .__import__ import Model
from .__import__ import Notifier
from .__import__ import build_notifier
from .__import__ import server_descriptor
from .Scopes import NOTIFIER_SCOPE


def build(near, mod_id = None) -> Notifier:
    if isinstance(near, Device):
        dev = near
        near_name = 'D%d' % dev.id
        far_name = 'L%d' % mod_id
        near_server = dev.host
    if isinstance(near, Model):
        mod = near
        dev = mod.host
        near_server = dev.host
        near_name = 'L%d' % mod.id
        far_name = 'D%d' % dev.id
    scope = NOTIFIER_SCOPE('BatchPipe/D_L')
    far_descriptor = server_descriptor(near_server)
    return build_notifier(near_server.h_notifier_context, scope, near_name, far_name, far_descriptor, True)
