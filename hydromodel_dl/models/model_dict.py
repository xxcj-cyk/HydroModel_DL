from torch.optim import Adam
from hydromodel_dl.models.lstm import SimpleLSTM
from hydromodel_dl.models.dpl4hbv import DplLstmHbv
from hydromodel_dl.models.dpl4xaj import DplLstmXaj
from hydromodel_dl.models.crits import (
    RMSELoss,
    RmseLoss,
    MultiOutLoss,
)

pytorch_model_dict = {
    "SimpleLSTM": SimpleLSTM,
    "DplLstmHbv": DplLstmHbv,
    "DplLstmXaj": DplLstmXaj,
}

pytorch_opt_dict = {"Adam": Adam}

pytorch_criterion_dict = {
    "RMSE": RMSELoss,
    # xxxSum means that calculate the criterion for each "feature"(the final dim of output), then sum them up
    "RMSESum": RmseLoss,
    "MultiOutLoss": MultiOutLoss,
}
