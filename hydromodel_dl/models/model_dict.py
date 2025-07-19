from torch.optim import Adam
from hydromodel_dl.models.lstm import SimpleLSTM, MultiLSTM
from hydromodel_dl.models.dpl4hbv import DplLstmHbv
from hydromodel_dl.models.dpl4xaj import DplLstmXaj
from hydromodel_dl.models.crits import (
    MAELoss,
    MSELoss,
    RMSELoss,
    PESLoss,
    HybridLoss,
    RMSEFloodLoss,
    HybridFloodLoss,
)

pytorch_model_dict = {
    "SimpleLSTM": SimpleLSTM,
    "MultiLSTM": MultiLSTM,
    "DplLstmHbv": DplLstmHbv,
    "DplLstmXaj": DplLstmXaj,
}

pytorch_opt_dict = {"Adam": Adam}

pytorch_criterion_dict = {
    "MAE": MAELoss,
    "MSE": MSELoss,
    "RMSE": RMSELoss,
    "PES": PESLoss,
    "Hybrid": HybridLoss,
    "RMSEFloodLoss": RMSEFloodLoss,
    "HybridFloodLoss": HybridFloodLoss,
}
