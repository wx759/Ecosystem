__all__ = ['NOTIFIER_SCOPE', 'NOTIFIER_SCOPE_ROOT']


from .__import__ import Server


def NOTIFIER_SCOPE_ROOT():
    return 'Notifier'


def NOTIFIER_SCOPE(sub_scope:str):
    return '%s/%s' % (NOTIFIER_SCOPE_ROOT(), sub_scope)