from torch.optim import Adam
from torchhydro_mini.models.lstm import SimpleLSTM, StandardLSTM
from torchhydro_mini.models.dpl4hbv import DplLstmHbv
from torchhydro_mini.models.crits import (
    RMSELoss,
    RmseLoss,
    MultiOutLoss,
)

pytorch_model_dict = {
    "SimpleLSTM": SimpleLSTM,
    "StandardLSTM": StandardLSTM,
    "DplLstmHbv": DplLstmHbv,
}

pytorch_opt_dict = {"Adam": Adam}

pytorch_criterion_dict = {
    "RMSE": RMSELoss,
    # xxxSum means that calculate the criterion for each "feature"(the final dim of output), then sum them up
    "RMSESum": RmseLoss,
    "MultiOutLoss": MultiOutLoss,
}
