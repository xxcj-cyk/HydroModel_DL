from abc import ABC, abstractmethod
from functools import reduce
from typing import Dict, Tuple

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader

from hydromodel_dl.datasets.data_dict import datasets_read_dict
from hydromodel_dl.datasets.data_sets import LongTermDataset
from hydromodel_dl.datasets.sampler import (
    data_sampler_dict,
)
from hydromodel_dl.models.model_dict import (
    pytorch_model_dict,
    pytorch_opt_dict,
    pytorch_criterion_dict,
)
from hydrodatautils.foundation.hydro_device import get_the_device
from hydromodel_dl.trainers.trainlogger import TrainLogger
from hydromodel_dl.trainers.train_utils import (
    EarlyStopper,
    denormalize4eval,
    evaluate_validation,
    compute_validation,
    model_infer,
    read_pth_from_model_loader,
    torch_single_train,
    _recover_samples_to_basin,
)


class DeepHydroInterface(ABC):
    """
    An abstract class used to handle different configurations
    of hydrological deep learning models + hyperparams for training, test, and predict functions.
    This class assumes that data is already split into test train and validation at this point.
    """

    def __init__(self, cfgs: Dict):
        """
        Parameters
        ----------
        cfgs
            configs for initializing DeepHydro
        """

        self._cfgs = cfgs

    @property
    def cfgs(self):
        """all configs"""
        return self._cfgs

    @property
    def weight_path(self):
        """weight path"""
        return self._cfgs["model_cfgs"]["weight_path"]

    @weight_path.setter
    def weight_path(self, weight_path):
        self._cfgs["model_cfgs"]["weight_path"] = weight_path

    @abstractmethod
    def load_model(self, mode="train") -> object:
        """Get a Hydro DL model"""
        raise NotImplementedError

    @abstractmethod
    def make_dataset(self, is_tra_val_te: str) -> object:
        """
        Initializes a pytorch dataset.

        Parameters
        ----------
        is_tra_val_te
            train or valid or test

        Returns
        -------
        object
            a dataset class loading data from data source
        """
        raise NotImplementedError

    @abstractmethod
    def model_train(self):
        """
        Train the model
        """
        raise NotImplementedError

    @abstractmethod
    def model_evaluate(self):
        """
        Evaluate the model
        """
        raise NotImplementedError


class DeepHydro(DeepHydroInterface):
    """
    The Base Trainer class for Hydrological Deep Learning models
    """

    def __init__(
        self,
        cfgs: Dict,
        pre_model=None,
    ):
        """
        Parameters
        ----------
        cfgs
            configs for the model
        pre_model
            a pre-trained model, if it is not None,
            we will use its weights to initialize this model
            by default None
        """
        super().__init__(cfgs)
        self.device_num = cfgs["training_cfgs"]["device"]
        self.device = get_the_device(self.device_num)
        self.pre_model = pre_model
        self.model = self.load_model()
        if cfgs["training_cfgs"]["train_mode"]:
            self.traindataset = self.make_dataset("train")
            if cfgs["data_cfgs"]["t_range_valid"] is not None:
                self.validdataset = self.make_dataset("valid")
        self.testdataset: LongTermDataset = self.make_dataset("test")
        print(f"Torch is using {str(self.device)}")

    def load_model(self, mode="train"):
        """
        Load a time series forecast model in pytorch_model_dict in model_dict_function.py

        Returns
        -------
        object
            model in pytorch_model_dict in model_dict_function.py
        """
        if mode == "infer":
            self.weight_path = self._get_trained_model()
        elif mode != "train":
            raise ValueError("Invalid mode; must be 'train' or 'infer'")
        model_cfgs = self.cfgs["model_cfgs"]
        model_name = model_cfgs["model_name"]
        if model_name not in pytorch_model_dict:
            raise NotImplementedError(
                f"Error the model {model_name} was not found in the model dict. Please add it."
            )
        if self.pre_model is not None:
            model = self._load_pretrain_model()
        elif self.weight_path is not None:
            # load model from pth file (saved weights and biases)
            model = self._load_model_from_pth()
        else:
            model = pytorch_model_dict[model_name](**model_cfgs["model_hyperparam"])
            # model_data = torch.load(weight_path)
            # model.load_state_dict(model_data)
        if torch.cuda.device_count() > 1 and len(self.device_num) > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            which_first_tensor = self.cfgs["training_cfgs"]["which_first_tensor"]
            sequece_first = which_first_tensor == "sequence"
            parallel_dim = 1 if sequece_first else 0
            model = nn.DataParallel(model, device_ids=self.device_num, dim=parallel_dim)
        model.to(self.device)
        return model

    def _load_pretrain_model(self):
        """load a pretrained model as the initial model"""
        return self.pre_model

    def _load_model_from_pth(self):
        weight_path = self.weight_path
        model_cfgs = self.cfgs["model_cfgs"]
        model_name = model_cfgs["model_name"]
        model = pytorch_model_dict[model_name](**model_cfgs["model_hyperparam"])
        checkpoint = torch.load(weight_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        print("Weights sucessfully loaded")
        return model

    def make_dataset(self, is_tra_val_te: str):
        """
        Initializes a pytorch dataset.

        Parameters
        ----------
        is_tra_val_te
            train or valid or test

        Returns
        -------
        object
            an object initializing from class in datasets_dirction_dict in data_dict.py
        """
        data_cfgs = self.cfgs["data_cfgs"]
        dataset_name = data_cfgs["dataset"]

        if dataset_name in list(datasets_read_dict.keys()):
            dataset = datasets_read_dict[dataset_name](data_cfgs, is_tra_val_te)
        else:
            raise NotImplementedError(
                f"Error the dataset {str(dataset_name)} was not found in the dataset dict. Please add it."
            )
        return dataset

    def model_train(self) -> None:
        """train a hydrological DL model"""
        # A dictionary of the necessary parameters for training
        training_cfgs = self.cfgs["training_cfgs"]
        # The file path to load model weights from; defaults to "model_save"
        model_filepath = self.cfgs["data_cfgs"]["test_path"]
        data_cfgs = self.cfgs["data_cfgs"]
        es = None
        if training_cfgs["early_stopping"]:
            es = EarlyStopper(training_cfgs["patience"])
        criterion = self._get_loss_func(training_cfgs)
        opt = self._get_optimizer(training_cfgs)
        scheduler = self._get_scheduler(training_cfgs, opt)
        max_epochs = training_cfgs["epochs"]
        start_epoch = training_cfgs["start_epoch"]
        # use PyTorch's DataLoader to load the data into batches in each epoch
        data_loader, validation_data_loader = self._get_dataloader(
            training_cfgs, data_cfgs
        )
        logger = TrainLogger(model_filepath, self.cfgs, opt)
        for epoch in range(start_epoch, max_epochs + 1):
            with logger.log_epoch_train(epoch) as train_logs:
                total_loss, n_iter_ep = torch_single_train(
                    self.model,
                    opt,
                    criterion,
                    data_loader,
                    device=self.device,
                    which_first_tensor=training_cfgs["which_first_tensor"],
                )
                train_logs["train_loss"] = total_loss
                train_logs["model"] = self.model

            valid_loss = None
            valid_metrics = None
            if data_cfgs["t_range_valid"] is not None:
                with logger.log_epoch_valid(epoch) as valid_logs:
                    valid_loss, valid_metrics = self._1epoch_valid(
                        training_cfgs, criterion, validation_data_loader, valid_logs
                    )

            self._scheduler_step(training_cfgs, scheduler, valid_loss)
            
            # Extract XAJ parameters if model is DplLstmXaj and extraction is enabled
            xaj_params = None
            extract_params = training_cfgs.get("extract_xaj_params", False)
            if extract_params and hasattr(self.model, 'pb_model') and hasattr(self.model.pb_model, 'params_names'):
                # This is a DplLstmXaj model, extract parameters grouped by basin
                try:
                    from hydromodel_dl.trainers.train_utils import _extract_xaj_params_by_basin
                    seq_first = training_cfgs["which_first_tensor"] != "batch"
                    # Extract parameters from all batches, grouped by basin
                    xaj_params = _extract_xaj_params_by_basin(
                        self.model, data_loader, self.device, seq_first
                    )
                except Exception as e:
                    print(f"Warning: Could not extract XAJ parameters: {e}")
                    xaj_params = None
            
            logger.save_session_param(
                epoch, total_loss, n_iter_ep, valid_loss, valid_metrics, xaj_params
            )
            logger.save_model_and_params(self.model, epoch, self.cfgs)
            if es and not es.check_loss(
                self.model,
                valid_loss,
                self.cfgs["data_cfgs"]["test_path"],
            ):
                print("Stopping model now")
                # Save training logs when early stopping occurs
                logger.save_training_logs(self.cfgs, self.model)
                break
        # logger.plot_model_structure(self.model)
        logger.tb.close()

        # return the trained model weights and bias and the epoch loss
        return self.model.state_dict(), sum(logger.epoch_loss) / len(logger.epoch_loss)

    def _get_scheduler(self, training_cfgs, opt):
        lr_scheduler_cfg = training_cfgs["lr_scheduler"]

        if "lr" in lr_scheduler_cfg and "lr_factor" not in lr_scheduler_cfg:
            scheduler = LambdaLR(opt, lr_lambda=lambda epoch: 1.0)
        elif isinstance(lr_scheduler_cfg, dict) and all(
            isinstance(epoch, int) for epoch in lr_scheduler_cfg
        ):
            scheduler = LambdaLR(
                opt, lr_lambda=lambda epoch: lr_scheduler_cfg.get(epoch, 1.0)
            )
        elif "lr_factor" in lr_scheduler_cfg and "lr_patience" not in lr_scheduler_cfg:
            scheduler = ExponentialLR(opt, gamma=lr_scheduler_cfg["lr_factor"])
        elif "lr_factor" in lr_scheduler_cfg:
            scheduler = ReduceLROnPlateau(
                opt,
                mode="min",
                factor=lr_scheduler_cfg["lr_factor"],
                patience=lr_scheduler_cfg["lr_patience"],
            )
        else:
            raise ValueError("Invalid lr_scheduler configuration")

        return scheduler

    def _scheduler_step(self, training_cfgs, scheduler, valid_loss):
        lr_scheduler_cfg = training_cfgs["lr_scheduler"]
        required_keys = {"lr_factor", "lr_patience"}
        if required_keys.issubset(lr_scheduler_cfg.keys()):
            scheduler.step(valid_loss)
        else:
            scheduler.step()

    def _1epoch_valid(
        self, training_cfgs, criterion, validation_data_loader, valid_logs
    ):
        valid_obss_np, valid_preds_np, valid_loss = compute_validation(
            self.model,
            criterion,
            validation_data_loader,
            device=self.device,
            which_first_tensor=training_cfgs["which_first_tensor"],
        )
        valid_logs["valid_loss"] = valid_loss
        if self.cfgs["evaluation_cfgs"]["calc_metrics"]:
            target_col = self.cfgs["data_cfgs"]["target_cols"]
            valid_metrics = evaluate_validation(
                validation_data_loader,
                valid_preds_np,
                valid_obss_np,
                self.cfgs["evaluation_cfgs"],
                target_col,
            )
            valid_logs["valid_metrics"] = valid_metrics
            return valid_loss, valid_metrics
        return valid_loss, None

    def _get_trained_model(self):
        model_loader = self.cfgs["evaluation_cfgs"]["model_loader"]
        model_pth_dir = self.cfgs["data_cfgs"]["test_path"]
        return read_pth_from_model_loader(model_loader, model_pth_dir)

    def model_evaluate(self) -> Tuple[Dict, np.array, np.array]:
        """
        A function to evaluate a model, called at end of training.

        Returns
        -------
        tuple[dict, np.array, np.array]
            eval_log, denormalized predictions and observations
        """
        self.model = self.load_model(mode="infer")
        preds_xr, obss_xr = self.inference()
        return preds_xr, obss_xr

    def inference(self) -> Tuple[xr.Dataset, xr.Dataset]:
        """infer using trained model and unnormalized results"""
        data_cfgs = self.cfgs["data_cfgs"]
        training_cfgs = self.cfgs["training_cfgs"]
        evaluation_cfgs = self.cfgs["evaluation_cfgs"]
        device = get_the_device(self.cfgs["training_cfgs"]["device"])
        test_dataloader = self._get_dataloader(training_cfgs, data_cfgs, mode="infer")
        seq_first = training_cfgs["which_first_tensor"] == "sequence"
        self.model.eval()
        # here the batch is just an index of lookup table, so any batch size could be chosen
        test_preds = []
        obss = []
        
        # Note: XAJ parameters are now only extracted and saved during training, not during inference
        # This is to keep only history and best parameter files
        
        with torch.no_grad():
            for batch_data in test_dataloader:
                # Handle different data formats: (xs, ys) or (xs, ys, event_ids)
                if len(batch_data) == 3:
                    xs, ys, event_ids = batch_data
                else:
                    xs, ys = batch_data
                
                # here the a batch doesn't mean a basin; it is only an index in lookup table
                # for NtoN mode, only basin is index in lookup table, so the batch is same as basin
                # for Nto1 mode, batch is only an index
                result = model_infer(seq_first, device, self.model, xs, ys, return_xaj_params=False)
                
                if len(result) == 3:  # Contains XAJ parameters (should not happen now)
                    ys, pred, _ = result
                else:
                    ys, pred = result
                
                test_preds.append(pred.cpu().numpy())
                obss.append(ys.cpu().numpy())
            pred = reduce(lambda x, y: np.vstack((x, y)), test_preds)
            obs = reduce(lambda x, y: np.vstack((x, y)), obss)
        if pred.ndim == 2:
            # TODO: check
            # the ndim is 2 meaning we use an Nto1 mode
            # as lookup table is (basin 1's all time length, basin 2's all time length, ...)
            # params of reshape should be (basin size, time length)
            pred = pred.flatten().reshape(test_dataloader.test_data.y.shape[0], -1, 1)
            obs = obs.flatten().reshape(test_dataloader.test_data.y.shape[0], -1, 1)

        if evaluation_cfgs["rolling"]:
            # TODO: now we only guarantee each time has only one value,
            # so we directly reshape the data rather than a real rolling
            ngrid = self.testdataset.ngrid
            nt = self.testdataset.nt
            target_len = len(data_cfgs["target_cols"])
            prec_window = data_cfgs["prec_window"]
            forecast_length = data_cfgs["forecast_length"]
            window_size = prec_window + forecast_length
            rho = data_cfgs["forecast_history"]
            recover_len = nt - rho + prec_window
            samples = int(pred.shape[0] / ngrid)
            pred_ = np.full((ngrid, recover_len, target_len), np.nan)
            obs_ = np.full((ngrid, recover_len, target_len), np.nan)
            # recover pred to pred_ and obs to obs_
            pred_4d = pred.reshape(ngrid, samples, window_size, target_len)
            obs_4d = obs.reshape(ngrid, samples, window_size, target_len)
            for i in range(ngrid):
                for j in range(recover_len - window_size + 1):
                    pred_[i, j : j + window_size, :] = pred_4d[i, j, :, :]
            for i in range(ngrid):
                for j in range(recover_len - window_size + 1):
                    obs_[i, j : j + window_size, :] = obs_4d[i, j, :, :]
            pred = pred_.reshape(ngrid, recover_len, target_len)
            obs = obs_.reshape(ngrid, recover_len, target_len)
        elif evaluation_cfgs["evaluator"]["eval_way"] == "1pace":
            # if we use 1pace, we need to recover the samples to basin
            pace_idx = evaluation_cfgs["evaluator"]["pace_idx"]
            pred = _recover_samples_to_basin(pred, test_dataloader, pace_idx)
            if (
                test_dataloader.dataset.name == "FloodEventDataset"
                or test_dataloader.dataset.name == "FloodEventDplDataset"
            ):
                # FloodEventDataset has two variables: streamflow and flood_event
                obs = obs[:, :, :-1]
            else:
                pass
            obs = _recover_samples_to_basin(obs, test_dataloader, pace_idx)
        pred_xr, obs_xr = denormalize4eval(
            test_dataloader, pred, obs, rolling=evaluation_cfgs["rolling"]
        )
        
        # Note: XAJ parameters are now only saved during training, not during inference
        # This is to keep only history and best parameter files
        
        return pred_xr, obs_xr

    # Removed _save_inference_xaj_params method - parameters are now only saved during training
    # to keep only history and best parameter files as requested

    def _get_optimizer(self, training_cfgs):
        params_in_opt = self.model.parameters()
        return pytorch_opt_dict[training_cfgs["optimizer"]](
            params_in_opt, **training_cfgs["optim_params"]
        )

    def _get_loss_func(self, training_cfgs):
        criterion_init_params = {}
        if "criterion_params" in training_cfgs:
            loss_param = training_cfgs["criterion_params"]
            if loss_param is not None:
                for key in loss_param.keys():
                    if key == "loss_funcs":
                        criterion_init_params[key] = pytorch_criterion_dict[
                            loss_param[key]
                        ]()
                    else:
                        criterion_init_params[key] = loss_param[key]
        return pytorch_criterion_dict[training_cfgs["criterion"]](
            **criterion_init_params
        )

    def _get_dataloader(self, training_cfgs, data_cfgs, mode="train"):
        if mode == "infer":
            ngrid = self.testdataset.ngrid
            if data_cfgs["sampler"] != "BasinBatchSampler":
                # TODO: this case should be tested more
                return DataLoader(
                    self.testdataset,
                    batch_size=training_cfgs["batch_size"],
                    shuffle=False,
                    sampler=None,
                    batch_sampler=None,
                    drop_last=False,
                    timeout=0,
                    worker_init_fn=None,
                )
            test_num_samples = self.testdataset.num_samples
            return DataLoader(
                self.testdataset,
                batch_size=test_num_samples // ngrid,
                shuffle=False,
                drop_last=False,
                timeout=0,
            )
        worker_num = 0
        pin_memory = False
        if "num_workers" in training_cfgs:
            worker_num = training_cfgs["num_workers"]
            print(f"using {str(worker_num)} workers")
        if "pin_memory" in training_cfgs:
            pin_memory = training_cfgs["pin_memory"]
            print(f"Pin memory set to {str(pin_memory)}")
        sampler = self._get_sampler(data_cfgs, self.traindataset)
        data_loader = DataLoader(
            self.traindataset,
            batch_size=training_cfgs["batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=worker_num,
            pin_memory=pin_memory,
            timeout=0,
        )
        if data_cfgs["t_range_valid"] is not None:
            validation_data_loader = DataLoader(
                self.validdataset,
                batch_size=training_cfgs["batch_size"],
                shuffle=False,
                num_workers=worker_num,
                pin_memory=pin_memory,
                timeout=0,
            )
            return data_loader, validation_data_loader

        return data_loader, None

    def _get_sampler(self, data_cfgs, train_dataset):
        """
        return data sampler based on the provided configuration and training dataset.

        Parameters
        ----------
        data_cfgs : dict
            Configuration dictionary containing parameters for data sampling. Expected keys are:
            - "batch_size": int, size of each batch.
            - "forecast_history": int, number of past time steps to consider.
            - "warmup_length": int, length of the warmup period.
            - "forecast_length": int, number of future time steps to predict.
            - "sampler": dict, containing:
            - "name": str, name of the sampler to use.
            - "sampler_hyperparam": dict, optional hyperparameters for the sampler.
        train_dataset : Dataset
            The training dataset object which contains the data to be sampled. Expected attributes are:
            - ngrid: int, number of grids in the dataset.
            - nt: int, number of time steps in the dataset.

        Returns
        -------
        sampler_class
            An instance of the specified sampler class, initialized with the provided dataset and hyperparameters.

        Raises
        ------
        NotImplementedError
            If the specified sampler name is not found in the `data_sampler_dict`.
        """
        if data_cfgs["sampler"] is None:
            return None
        batch_size = data_cfgs["batch_size"]
        rho = data_cfgs["forecast_history"]
        warmup_length = data_cfgs["warmup_length"]
        horizon = data_cfgs["forecast_length"]
        ngrid = train_dataset.ngrid
        nt = train_dataset.nt
        sampler_name = data_cfgs["sampler"]
        if sampler_name not in data_sampler_dict:
            raise NotImplementedError(f"Sampler {sampler_name} not implemented yet")
        sampler_class = data_sampler_dict[sampler_name]
        sampler_hyperparam = {}
        if sampler_name == "KuaiSampler":
            sampler_hyperparam |= {
                "batch_size": batch_size,
                "warmup_length": warmup_length,
                "rho_horizon": rho + horizon,
                "ngrid": ngrid,
                "nt": nt,
            }
        return sampler_class(train_dataset, **sampler_hyperparam)


class TransLearnHydro(DeepHydro):
    def __init__(self, cfgs: Dict, pre_model=None):
        super().__init__(cfgs, pre_model)

    def load_model(self, mode="train"):
        """Load model for transfer learning"""
        model_cfgs = self.cfgs["model_cfgs"]
        if self.weight_path is None and self.pre_model is None:
            raise NotImplementedError(
                "For transfer learning, we need a pre-trained model"
            )
        model = super().load_model(mode)
        if (
            "weight_path_add" in model_cfgs
            and "freeze_params" in model_cfgs["weight_path_add"]
        ):
            freeze_params = model_cfgs["weight_path_add"]["freeze_params"]
            for param in freeze_params:
                exec(f"model.{param}.requires_grad = False")
        return model

    def _load_model_from_pth(self):
        weight_path = self.weight_path
        model_cfgs = self.cfgs["model_cfgs"]
        model_name = model_cfgs["model_name"]
        model = pytorch_model_dict[model_name](**model_cfgs["model_hyperparam"])
        checkpoint = torch.load(weight_path, map_location=self.device)
        if "weight_path_add" in model_cfgs:
            if "excluded_layers" in model_cfgs["weight_path_add"]:
                # delete some layers from source model if we don't need them
                excluded_layers = model_cfgs["weight_path_add"]["excluded_layers"]
                for layer in excluded_layers:
                    del checkpoint[layer]
                print("sucessfully deleted layers")
            else:
                print("directly loading identically-named layers of source model")
        model.load_state_dict(checkpoint, strict=False)
        print("Weights sucessfully loaded")
        return model


class MultiTaskHydro(DeepHydro):
    def __init__(self, cfgs: Dict, pre_model=None):
        super().__init__(cfgs, pre_model)

    def _get_optimizer(self, training_cfgs):
        params_in_opt = self.model.parameters()
        if training_cfgs["criterion"] == "UncertaintyWeights":
            # log_var = torch.zeros((1,), requires_grad=True)
            log_vars = [
                torch.zeros((1,), requires_grad=True, device=self.device)
                for _ in range(training_cfgs["multi_targets"])
            ]
            params_in_opt = list(self.model.parameters()) + log_vars
        return pytorch_opt_dict[training_cfgs["optimizer"]](
            params_in_opt, **training_cfgs["optim_params"]
        )

    def _get_loss_func(self, training_cfgs):
        if "criterion_params" in training_cfgs:
            loss_param = training_cfgs["criterion_params"]
            if loss_param is not None:
                criterion_init_params = {
                    key: (
                        pytorch_criterion_dict[loss_param[key]]()
                        if key == "loss_funcs"
                        else loss_param[key]
                    )
                    for key in loss_param.keys()
                }
        if training_cfgs["criterion"] == "MultiOutWaterBalanceLoss":
            # TODO: hard code for streamflow and ET
            stat_dict = self.traindataset.target_scaler.stat_dict
            stat_dict_keys = list(stat_dict.keys())
            q_name = np.intersect1d(
                [
                    "usgsFlow",
                    "streamflow",
                    "Q",
                    "qobs",
                ],
                stat_dict_keys,
            )[0]
            et_name = np.intersect1d(
                [
                    "ET",
                    "LE",
                    "GPP",
                    "Ec",
                    "Es",
                    "Ei",
                    "ET_water",
                    # sum pf ET components in PML V2
                    "ET_sum",
                ],
                stat_dict_keys,
            )[0]
            q_mean = self.training.target_scaler.stat_dict[q_name][2]
            q_std = self.training.target_scaler.stat_dict[q_name][3]
            et_mean = self.training.target_scaler.stat_dict[et_name][2]
            et_std = self.training.target_scaler.stat_dict[et_name][3]
            means = [q_mean, et_mean]
            stds = [q_std, et_std]
            criterion_init_params["means"] = means
            criterion_init_params["stds"] = stds
        return pytorch_criterion_dict[training_cfgs["criterion"]](
            **criterion_init_params
        )


model_type_dict = {
    "Normal": DeepHydro,
    "TransLearn": TransLearnHydro,
    "MTL": MultiTaskHydro,
}
