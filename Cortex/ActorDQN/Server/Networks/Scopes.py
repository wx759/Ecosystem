

from .__import__ import GLOBAL_SCOPE_RUNTIME_PARAS
from .__import__ import GLOBAL_SCOPE_SHADOW


def ROOT_SCOPE_SERVER(serv_id): return 'Server%d/' % serv_id
def ROOT_SCOPE_DEVICE(dev_id): return 'Device%d/' % dev_id
def ROOT_SCOPE_MODEL(mod_id): return 'Model%d/' % mod_id

GLOABL_SCOPE_BATCH_PIPE = 'Batch/'
GLOBAL_SCOPE_Q_NET = 'Q_Net/'
GLOBAL_SCOPE_R_NET = 'R_Net/'
GLOBAL_SCOPE_TRAINED_STEP = 'Trained_Step/'
GLOBAL_SCOPE_Q_LEARNER = 'Q_Learner/'