from torchhydro.models.dpl4hbv import DplLstmHbv, DplAnnHbv
from torchhydro.models.cudnnlstm import (
    CudnnLstmModel,
    SimpleLSTM,
    CudnnLstmModelMultiOutput,
)

from torchhydro.models.lstm import SimpleLSTMForecast
from torch.optim import Adam, SGD, Adadelta
from torchhydro.models.crits import (
    RMSELoss,
    RmseLoss,
    MultiOutLoss,

)
from torchhydro.models.dpl4xaj import DplLstmXaj
from torchhydro.models.dpl4hbv import DplLstmHbv
from torchhydro.models.dpl4gr4j import DplLstmGr4j

"""
Utility dictionaries to map a string to a class.
"""
pytorch_model_dict = {
    "SimpleLSTM": SimpleLSTM,
    "KuaiLSTM": CudnnLstmModel,
    "KuaiLSTMMultiOut": CudnnLstmModelMultiOutput,
    "DplLstmHbv": DplLstmHbv,
    "DplLstmGr4j": DplLstmGr4j,
    "DplLstmXaj": DplLstmXaj,
}

pytorch_criterion_dict = {
    "RMSE": RMSELoss,
    # xxxSum means that calculate the criterion for each "feature"(the final dim of output), then sum them up
    "RMSESum": RmseLoss,
    "MultiOutLoss": MultiOutLoss,
}

pytorch_opt_dict = {"Adam": Adam, "SGD": SGD, "Adadelta": Adadelta}
