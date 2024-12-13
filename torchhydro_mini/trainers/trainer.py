import random
import numpy as np
from typing import Dict
import torch
from trainers.deep_hydro import model_type_dict
from trainers.resulter import Resulter


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_evaluate(cfgs: Dict):
    # set random seed
    random_seed = cfgs["training_cfgs"]["random_seed"]
    set_random_seed(random_seed)
    # Initialize a Result Handler to manage saving and evaluating results
    resulter = Resulter(cfgs)
    # Create a Deep Hydromodel instance
    deephydro = _get_deep_hydro(cfgs)
    # Train the model in different modes (training or skip training or continue training)
    if cfgs["training_cfgs"]["train_mode"] and (
        (
            deephydro.weight_path is not None
            and deephydro.cfgs["model_cfgs"]["continue_train"]
        )
        or (deephydro.weight_path is None)
    ):
        deephydro.model_train()
    # Evaluate the model
    preds, obss = deephydro.model_evaluate()
    # Save the model configuration
    resulter.save_cfg(deephydro.cfgs)
    # Save the evaluation results
    resulter.save_result(preds, obss)
    # Compute and save evaluation metrics
    resulter.eval_result(preds, obss)


def _get_deep_hydro(cfgs):
    model_type = cfgs["model_cfgs"]["model_type"]
    return model_type_dict[model_type](cfgs)