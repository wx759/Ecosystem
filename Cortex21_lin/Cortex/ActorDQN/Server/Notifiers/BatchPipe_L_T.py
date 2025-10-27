__all__ = ['build']


from .__import__ import Model
from .__import__ import Notifier
from .__import__ import build_notifier
from .__import__ import server_descriptor
from .Scopes import NOTIFIER_SCOPE


def build(near:Model, L_side:bool) -> Notifier:
    mod = near
    dev = mod.host
    serv = dev.host
    scope = NOTIFIER_SCOPE('BatchPipe/L_T')
    far_descriptor = server_descriptor(serv)
    if L_side:
        near_name = 'L%dD%d' % (mod.id, dev.id)
        far_name = 'T%dD%d' % (mod.id, dev.id)
    else:
        near_name = 'T%dD%d' % (mod.id, dev.id)
        far_name = 'L%dD%d' % (mod.id, dev.id)
    return build_notifier(serv.h_notifier_context, scope, near_name, far_name, far_descriptor, True)
