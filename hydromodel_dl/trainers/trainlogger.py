from contextlib import contextmanager
from datetime import datetime
import json
import os
import time
from hydrodatautils.foundation import hydro_format
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from hydromodel_dl.trainers.train_utils import get_lastest_logger_file_in_a_dir


def _make_json_serializable(obj):
    """
    Convert numpy arrays, PyTorch tensors and other non-JSON-serializable objects to JSON-serializable format
    
    Parameters
    ----------
    obj : any
        Object to convert
        
    Returns
    -------
    any
        JSON-serializable version of the object
    """
    import numpy as np
    import torch
    
    if isinstance(obj, torch.Tensor):
        # Convert PyTorch tensor to numpy array first, then to list
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj


def save_model(model, model_file, gpu_num=1):
    try:
        if torch.cuda.device_count() > 1 and gpu_num > 1:
            torch.save(model.module.state_dict(), model_file)
        else:
            torch.save(model.state_dict(), model_file)
    except RuntimeError:
        torch.save(model.module.state_dict(), model_file)


def save_model_params_log(params, params_log_path):
    time_stamp = datetime.now().strftime("%d_%B_%Y%I_%M%p")
    params_log_file = os.path.join(params_log_path, f"{time_stamp}.json")
    hydro_format.serialize_json(params, params_log_file)


class TrainLogger:
    def __init__(self, model_filepath, params, opt):
        self.training_cfgs = params["training_cfgs"]
        self.data_cfgs = params["data_cfgs"]
        self.evaluation_cfgs = params["evaluation_cfgs"]
        self.model_cfgs = params["model_cfgs"]
        self.opt = opt
        self.training_save_dir = model_filepath
        self.tb = SummaryWriter(self.training_save_dir)
        self.session_params = []
        self.train_time = []
        # log loss for each epoch
        self.epoch_loss = []
        # XAJ parameters tracking
        self.xaj_params_history = []
        self.best_xaj_params = None
        self.best_loss = float('inf')
        self.best_epoch = 0
        # reload previous logs if continue_train is True and weight_path is not None
        if (
            self.model_cfgs["continue_train"]
            and self.model_cfgs["weight_path"] is not None
        ):
            the_logger_file = get_lastest_logger_file_in_a_dir(self.training_save_dir)
            if the_logger_file is not None:
                with open(the_logger_file, "r") as f:
                    logs = json.load(f)
                start_epoch = self.training_cfgs["start_epoch"]
                # read the logs before start_epoch and load them to session_params, train_time, epoch_loss
                for log in logs["run"]:
                    if log["epoch"] < start_epoch:
                        self.session_params.append(log)
                        self.train_time.append(log["train_time"])
                        self.epoch_loss.append(float(log["train_loss"]))

    def save_session_param(
        self, epoch, total_loss, n_iter_ep, valid_loss=None, valid_metrics=None, xaj_params=None
    ):
        if valid_loss is None or valid_metrics is None:
            epoch_params = {
                "epoch": epoch,
                "train_loss": str(total_loss),
                "iter_num": n_iter_ep,
            }
        else:
            epoch_params = {
                "epoch": epoch,
                "train_loss": str(total_loss),
                "validation_loss": str(valid_loss),
                "validation_metric": valid_metrics,
                "iter_num": n_iter_ep,
            }
            
        epoch_params["train_time"] = self.train_time[epoch - 1]
        self.session_params.append(epoch_params)
        
        # Save XAJ parameters separately if provided
        if xaj_params is not None:
            self.save_xaj_params(epoch, total_loss, valid_loss, xaj_params)

    def save_xaj_params(self, epoch, train_loss, valid_loss, xaj_params):
        """
        Save XAJ parameters separately from training logs
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        train_loss : float
            Training loss for this epoch
        valid_loss : float or None
            Validation loss for this epoch
        xaj_params : dict
            XAJ parameters dictionary, can be either:
            - Old format: {param_name: [value], ...} (single set of parameters)
            - New format: {basin_id: {param_name: [value], ...}, ...} (parameters by basin)
        """
        # Ensure all values are JSON serializable
        serialized_params = _make_json_serializable(xaj_params)
        
        # Create parameter record
        param_record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "validation_loss": float(valid_loss) if valid_loss is not None else None,
            "parameters": serialized_params
        }
        
        # Add to history
        self.xaj_params_history.append(param_record)
        
        # Check if this is the best epoch (lowest validation loss)
        current_loss = valid_loss if valid_loss is not None else train_loss
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.best_xaj_params = param_record.copy()
            print(f"New best parameters found at epoch {epoch} with loss {current_loss:.6f}")

    @contextmanager
    def log_epoch_train(self, epoch):
        start_time = time.time()
        logs = {}
        # here content in the 'with' block will be performed after yeild
        yield logs
        total_loss = logs["train_loss"]
        elapsed_time = time.time() - start_time
        lr = self.opt.param_groups[0]["lr"]
        log_str = "Epoch {} Loss {:.4f} time {:.2f} lr {}".format(
            epoch, total_loss, elapsed_time, lr
        )
        print(log_str)
        model = logs["model"]
        print(model)
        self.tb.add_scalar("Loss", total_loss, epoch)
        # self.plot_hist_img(model, epoch)
        self.train_time.append(log_str)
        self.epoch_loss.append(total_loss)

    @contextmanager
    def log_epoch_valid(self, epoch):
        logs = {}
        yield logs
        valid_loss = logs["valid_loss"]
        if self.evaluation_cfgs["calc_metrics"]:
            valid_metrics = logs["valid_metrics"]
            val_log = "Epoch {} Valid Loss {:.4f} Valid Metric {}".format(
                epoch, valid_loss, valid_metrics
            )
            print(val_log)
            self.tb.add_scalar("ValidLoss", valid_loss, epoch)
            target_col = self.data_cfgs["target_cols"]
            if self.data_cfgs["dataset"] == "FloodEventDplDataset":
                target_col = target_col[:-1]
            evaluation_metrics = self.evaluation_cfgs["metrics"]
            for i in range(len(target_col)):
                for evaluation_metric in evaluation_metrics:
                    self.tb.add_scalar(
                        f"Valid{target_col[i]}{evaluation_metric}mean",
                        np.mean(
                            valid_metrics[f"{evaluation_metric} of {target_col[i]}"]
                        ),
                        epoch,
                    )
                    self.tb.add_scalar(
                        f"Valid{target_col[i]}{evaluation_metric}median",
                        np.median(
                            valid_metrics[f"{evaluation_metric} of {target_col[i]}"]
                        ),
                        epoch,
                    )
        else:
            val_log = "Epoch {} Valid Loss {:.4f} ".format(epoch, valid_loss)
            print(val_log)
            self.tb.add_scalar("ValidLoss", valid_loss, epoch)

    def save_model_and_params(self, model, epoch, params):
        final_epoch = params["training_cfgs"]["epochs"]
        save_epoch = params["training_cfgs"]["save_epoch"]
        if save_epoch is None or save_epoch == 0 and epoch != final_epoch:
            return
        if (save_epoch > 0 and epoch % save_epoch == 0) or epoch == final_epoch:
            # save for save_epoch
            model_file = os.path.join(
                self.training_save_dir, f"model_Ep{str(epoch)}.pth"
            )
            save_model(model, model_file)
        if epoch == final_epoch:
            self._save_final_epoch(params, model)

    def save_training_logs(self, params, model):
        """
        Save training logs to JSON files. This method can be called when early stopping occurs
        or at the final epoch.
        
        Parameters
        ----------
        params : dict
            Configuration parameters
        model : torch.nn.Module
            The trained model
        """
        final_path = params["data_cfgs"]["test_path"]
        params["run"] = self.session_params
        
        # Make sure all parameters are JSON serializable
        params_serializable = _make_json_serializable(params)
        
        time_stamp = datetime.now().strftime("%d_%B_%Y%I_%M%p")
        model_save_path = os.path.join(final_path, f"{time_stamp}_model.pth")
        save_model(model, model_save_path)
        save_model_params_log(params_serializable, final_path)
        # also save one for a training directory for one hyperparameter setting
        save_model_params_log(params_serializable, self.training_save_dir)
        
        # Save XAJ parameters separately only if model is XAJ model (e.g., DplLstmXaj)
        # Check if model has pb_model attribute which indicates it's an XAJ model
        if hasattr(model, 'pb_model') and hasattr(model.pb_model, 'params_names'):
            self._save_xaj_params_files(final_path, time_stamp)

    def _save_final_epoch(self, params, model):
        # In final epoch, we save the model and params in test_path
        self.save_training_logs(params, model)

    def _save_xaj_params_files(self, final_path, time_stamp):
        """
        Save XAJ parameters to separate JSON files
        
        Parameters
        ----------
        final_path : str
            Path where to save the files
        time_stamp : str
            Timestamp for file naming
        """
        # Save parameter history (all epochs)
        # Even if history is empty, save an empty file to indicate training completed
        history_file = os.path.join(final_path, f"{time_stamp}_xaj_params_history.json")
        if self.xaj_params_history:
            # Determine parameter names from first epoch
            first_params = self.xaj_params_history[0]["parameters"]
            param_names = []
            if isinstance(first_params, dict) and first_params:
                # Check if it's the new format (by basin) or old format
                first_value = list(first_params.values())[0]
                if isinstance(first_value, dict):
                    # New format: {basin_id: {param_name: [value], ...}, ...}
                    # Get parameter names from first basin
                    first_basin = list(first_params.keys())[0]
                    param_names = list(first_params[first_basin].keys())
                else:
                    # Old format: {param_name: [value], ...}
                    param_names = list(first_params.keys())
            
            history_data = {
                "summary": {
                    "total_epochs": len(self.xaj_params_history),
                    "best_epoch": int(self.best_epoch) if self.best_epoch > 0 else None,
                    "best_loss": float(self.best_loss) if self.best_loss != float('inf') else None,
                    "parameter_names": param_names
                },
                "history": _make_json_serializable(self.xaj_params_history)
            }
        else:
            print("Warning: No XAJ parameters history found, saving empty history file")
            history_data = {
                "summary": {
                    "total_epochs": 0,
                    "best_epoch": None,
                    "best_loss": None,
                    "parameter_names": []
                },
                "history": []
            }
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        print(f"XAJ parameters history saved to: {history_file}")
        
        # Save best parameters
        if self.best_xaj_params is not None:
            best_file = os.path.join(final_path, f"{time_stamp}_xaj_params_best.json")
            # Determine parameter names from best parameters
            best_params = self.best_xaj_params["parameters"]
            param_names = []
            if isinstance(best_params, dict) and best_params:
                # Check if it's the new format (by basin) or old format
                first_value = list(best_params.values())[0]
                if isinstance(first_value, dict):
                    # New format: {basin_id: {param_name: [value], ...}, ...}
                    first_basin = list(best_params.keys())[0]
                    param_names = list(best_params[first_basin].keys())
                else:
                    # Old format: {param_name: [value], ...}
                    param_names = list(best_params.keys())
            
            best_data = {
                "summary": {
                    "best_epoch": int(self.best_epoch),
                    "best_loss": float(self.best_loss),
                    "parameter_names": param_names
                },
                "best_parameters": _make_json_serializable(self.best_xaj_params)
            }
            
            with open(best_file, 'w', encoding='utf-8') as f:
                json.dump(best_data, f, indent=2, ensure_ascii=False)
            print(f"Best XAJ parameters saved to: {best_file}")
        else:
            print("Warning: No best XAJ parameters found (best_xaj_params is None)")
        
        # Also save copies in training directory
        training_history_file = os.path.join(self.training_save_dir, f"{time_stamp}_xaj_params_history.json")
        with open(training_history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
            
        if self.best_xaj_params is not None:
            training_best_file = os.path.join(self.training_save_dir, f"{time_stamp}_xaj_params_best.json")
            with open(training_best_file, 'w', encoding='utf-8') as f:
                json.dump(best_data, f, indent=2, ensure_ascii=False)

    def plot_hist_img(self, model, global_step):
        for tag, parm in model.named_parameters():
            self.tb.add_histogram(
                f"{tag}_hist", parm.detach().cpu().numpy(), global_step
            )
            if len(parm.shape) == 2:
                img_format = "HW"
                if parm.shape[0] > parm.shape[1]:
                    img_format = "WH"
                    self.tb.add_image(
                        f"{tag}_img",
                        parm.detach().cpu().numpy(),
                        global_step,
                        dataformats=img_format,
                    )

    def plot_model_structure(self, model):
        """plot model structure in tensorboard

        Parameters
        ----------
        model :
            torch model
        """
        # input4modelplot = torch.randn(
        #     self.data_cfgs["batch_size"],
        #     self.data_cfgs["forecast_history"],
        #     # self.model_cfgs["model_hyperparam"]["n_input_features"],
        #     self.model_cfgs["model_hyperparam"]["input_size"],
        # )
        if self.data_cfgs["model_mode"] == "single":
            input4modelplot = [
                torch.randn(
                    self.data_cfgs["batch_size"],
                    self.data_cfgs["forecast_history"],
                    self.data_cfgs["input_features"] - 1,
                ),
                torch.randn(
                    self.data_cfgs["batch_size"],
                    self.data_cfgs["forecast_history"],
                    self.data_cfgs["cnn_size"],
                ),
                torch.rand(
                    self.data_cfgs["batch_size"], 1, self.data_cfgs["output_features"]
                ),
            ]
        else:
            input4modelplot = [
                torch.randn(
                    self.data_cfgs["batch_size"],
                    self.data_cfgs["forecast_history"],
                    self.data_cfgs["input_features"],
                ),
                torch.randn(
                    self.data_cfgs["batch_size"],
                    self.data_cfgs["forecast_history"],
                    self.data_cfgs["input_size_encoder2"],
                ),
                torch.rand(
                    self.data_cfgs["batch_size"], 1, self.data_cfgs["output_features"]
                ),
            ]
        self.tb.add_graph(model, input4modelplot)
