from torch.optim import Adam
from hydromodel_dl.models.lstm import SimpleLSTM
from hydromodel_dl.models.dpl4hbv import DplLstmHbv
from hydromodel_dl.models.dpl4xaj import DplLstmXaj
from hydromodel_dl.models.crits import (
    MultiOutLoss,
    MAELoss,
    MSELoss,
    RMSELoss,    
    PESLoss,
    HybridLoss,
)

pytorch_model_dict = {
    "SimpleLSTM": SimpleLSTM,
    "DplLstmHbv": DplLstmHbv,
    "DplLstmXaj": DplLstmXaj,
}

pytorch_opt_dict = {"Adam": Adam}

pytorch_criterion_dict = {
    "MultiOutLoss": MultiOutLoss,
    "MAE": MAELoss,
    "MSE": MSELoss,
    "RMSE": RMSELoss,
    "PES": PESLoss,
    "Hybrid": HybridLoss,
}
