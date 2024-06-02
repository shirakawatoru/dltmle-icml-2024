from .LAY.dltmle import DeepLTMLE
from .LAY.dltmle2 import DeepLTMLE2
from .LAY.deepace import DeepACE

def load_model(model_name, hparams, data):
    if (model_name == 'dltmle'):
        return DeepLTMLE(data.dim_static, data.dim_dynamic, data.tau, **hparams)
    elif (model_name == 'dltmle2'):
        return DeepLTMLE2(data.dim_static, data.dim_dynamic, data.tau, **hparams)
    elif (model_name == 'deepace'):
        return DeepACE(data.dim_static, data.dim_dynamic, data.tau, **hparams)
    else:
        raise ValueError('model_name not recognized: {}'.format(model_name))