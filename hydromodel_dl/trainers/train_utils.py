import copy
import fnmatch
import os
import re
import shutil
import dask
from functools import reduce
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import xarray as xr
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from hydrodatautils.foundation.hydro_statistic import statistic_nd_error
from hydrodatautils.foundation.hydro_model import (
    get_lastest_file_in_a_dir,
    get_latest_file_in_a_lst,
)
from hydrodatautils.foundation.hydro_format import unserialize_json
from hydromodel_dl.models.crits import (
    GaussianLoss,
    RMSEFloodSampleLoss,
    RMSEFloodEventLoss,
    PeakFocusedFloodSampleLoss,
    PeakFocusedFloodEventLoss,
)


def _extract_xaj_params(model, xs, device):
    """
    Extract XAJ parameters from DplLstmXaj model
    
    Parameters
    ----------
    model : DplLstmXaj
        The DplLstmXaj model
    xs : list or tensor
        Input data
    device : torch.device
        Device where tensors are located
        
    Returns
    -------
    dict
        Dictionary containing XAJ parameter names and their denormalized values
    """
    import torch.nn.functional as F
    
    # Get the normalized input (z) - assuming it's the second input for DplLstmXaj
    if len(xs) >= 2:
        z = xs[1]  # normalized input for DL model
    else:
        # If only one input, use it as both x and z
        z = xs[0]
    
    # Generate parameters using LSTM
    gen = model.dl_model(z)
    
    # Apply parameter limiting function
    if model.param_func == "sigmoid":
        params_ = F.sigmoid(gen)
    elif model.param_func == "clamp":
        params_ = torch.clamp(gen, min=0.0, max=1.0)
    else:
        raise NotImplementedError(f"Parameter function {model.param_func} not supported")
    
    # Get final time step parameters
    params = params_[-1, :, :]  # [batch_size, n_params]
    
    # Denormalize parameters using the same logic as in Xaj4Dpl.forward()
    pb_model = model.pb_model
    denormalized_params = {}
    
    # Extract each parameter and denormalize
    param_names = pb_model.params_names
    param_scales = {
        'K': pb_model.k_scale,
        'B': pb_model.b_scale,
        'IM': pb_model.im_sacle,
        'UM': pb_model.um_scale,
        'LM': pb_model.lm_scale,
        'DM': pb_model.dm_scale,
        'C': pb_model.c_scale,
        'SM': pb_model.sm_scale,
        'EX': pb_model.ex_scale,
        'KI': pb_model.ki_scale,
        'KG': pb_model.kg_scale,
        'A': pb_model.a_scale,
        'THETA': pb_model.theta_scale,
        'CI': pb_model.ci_scale,
        'CG': pb_model.cg_scale,
    }
    
    for i, param_name in enumerate(param_names):
        if param_name in param_scales:
            scale = param_scales[param_name]
            denormalized_value = scale[0] + params[:, i] * (scale[1] - scale[0])
            # Convert to Python list for JSON serialization
            denormalized_params[param_name] = denormalized_value.detach().cpu().numpy().tolist()
    
    return denormalized_params


def _extract_xaj_params_by_basin(model, data_loader, device, seq_first):
    """
    Extract XAJ parameters from DplLstmXaj model, grouped by basin
    
    This function traverses all batches in the data loader and groups parameters by basin.
    Each basin's parameters are averaged across all samples from that basin.
    
    Parameters
    ----------
    model : DplLstmXaj
        The DplLstmXaj model
    data_loader : DataLoader
        Data loader to iterate through
    device : torch.device
        Device where tensors are located
    seq_first : bool
        Whether input data is sequence-first format
        
    Returns
    -------
    dict
        Dictionary with basin IDs as keys and parameter dictionaries as values
        Format: {basin_id: {param_name: [value], ...}, ...}
    """
    from collections import defaultdict
    import torch.nn.functional as F
    
    # Group parameters by basin
    params_by_basin = defaultdict(lambda: defaultdict(list))
    
    # Set model to eval mode and disable gradient computation
    model.eval()
    with torch.no_grad():
        for batch_data in data_loader:
            # Handle different data formats: (src, trg) or (src, trg, event_ids)
            if len(batch_data) == 3:
                src, trg, event_ids = batch_data
            else:
                src, trg = batch_data
                event_ids = None
            
            # Skip if no event_ids (basin IDs)
            if event_ids is None:
                continue
            
            # Prepare inputs
            if isinstance(src, (list, tuple)) and len(src) >= 2:
                sample_src = src
            else:
                sample_src = [src]
            
            # Convert event_ids to list if needed
            if isinstance(event_ids, torch.Tensor):
                event_ids = event_ids.tolist()
            elif not isinstance(event_ids, list):
                event_ids = list(event_ids)
            
            # Extract parameters for this batch
            try:
                result = model_infer(seq_first, device, model, sample_src, trg, return_xaj_params=True)
                if len(result) == 3:
                    _, _, xaj_params_raw = result
                    
                    # Group parameters by basin
                    for i, basin_id in enumerate(event_ids):
                        basin_id_str = str(basin_id)
                        for param_name, param_values in xaj_params_raw.items():
                            if isinstance(param_values, list) and i < len(param_values):
                                params_by_basin[basin_id_str][param_name].append(param_values[i])
            except Exception as e:
                print(f"Warning: Could not extract XAJ parameters from batch: {e}")
                continue
    
    # Average parameters for each basin
    params_by_basin_averaged = {}
    for basin_id, param_dict in params_by_basin.items():
        params_by_basin_averaged[basin_id] = {}
        for param_name, param_values in param_dict.items():
            if param_values:
                # Calculate mean across all samples from this basin
                params_by_basin_averaged[basin_id][param_name] = [
                    sum(param_values) / len(param_values)
                ]
            else:
                params_by_basin_averaged[basin_id][param_name] = []
    
    # Sort basins by ID before returning
    def _basin_sort_key(basin_id):
        """Sort key for basin IDs - handles both numeric and string IDs"""
        try:
            # Try to convert to int for numeric sorting
            return (0, int(basin_id))
        except (ValueError, TypeError):
            # If not numeric, sort as string
            return (1, str(basin_id))
    
    sorted_basin_ids = sorted(params_by_basin_averaged.keys(), key=_basin_sort_key)
    return {basin_id: params_by_basin_averaged[basin_id] for basin_id in sorted_basin_ids}


def model_infer(seq_first, device, model, xs, ys, return_xaj_params=False):
    """_summary_

    Parameters
    ----------
    seq_first : bool
        if True, the input data is sequence first
    device : torch.device
        cpu or gpu
    model : torch.nn.Module
        the model
    xs : list or tensor
        xs is always batch first
    ys : tensor
        observed data
    return_xaj_params : bool
        if True and model is DplLstmXaj, also return XAJ parameters

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor] or tuple[torch.Tensor, torch.Tensor, dict]
        first is the observed data, second is the predicted data;
        if return_xaj_params=True and model is DplLstmXaj, third is dict of XAJ parameters;
        both tensors are batch first
    """
    if type(xs) is list:
        xs = [
            (
                data_tmp.permute([1, 0, 2]).to(device)
                if seq_first and data_tmp.ndim == 3
                else data_tmp.to(device)
            )
            for data_tmp in xs
        ]
    else:
        xs = [
            (
                xs.permute([1, 0, 2]).to(device)
                if seq_first and xs.ndim == 3
                else xs.to(device)
            )
        ]
    ys = (
        ys.permute([1, 0, 2]).to(device)
        if seq_first and ys.ndim == 3
        else ys.to(device)
    )
    output = model(*xs)
    if type(output) is tuple:
        # Convention: y_p must be the first output of model
        output = output[0]
    
    # Check if we need to extract XAJ parameters for DplLstmXaj model
    xaj_params = None
    if return_xaj_params and hasattr(model, 'pb_model') and hasattr(model.pb_model, 'params_names'):
        # This is a DplLstmXaj model, extract the current XAJ parameters
        xaj_params = _extract_xaj_params(model, xs, device)
    
    if seq_first:
        output = output.transpose(0, 1)
        ys = ys.transpose(0, 1)
    
    if xaj_params is not None:
        return ys, output, xaj_params
    else:
        return ys, output


def denormalize4eval(eval_dataloader, output, labels, rolling=False):
    """_summary_

    Parameters
    ----------
    eval_dataloader : _type_
        dataloader for validation or test
    output : np.ndarray
        batch-first model output
    labels : np.ndarray
        batch-first observed data
    rolling: bool
        if True, to guarantee each time has only one value for one variable of a sample
        we just cut the data.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        predicted data and observed data
    """
    target_scaler = eval_dataloader.dataset.target_scaler
    target_data = target_scaler.data_target
    # the units are dimensionless for pure DL models
    units = {k: "dimensionless" for k in target_data.attrs["units"].keys()}
    if target_scaler.pbm_norm:
        units = {**units, **target_data.attrs["units"]}
    if rolling:
        prec_window = target_scaler.data_cfgs["prec_window"]
        rho = target_scaler.data_cfgs["forecast_history"]
        # TODO: -1 because seq2seqdataset has one more time, hence we need to cut it, as rolling will be deprecated, we don't modify it yet
        selected_time_points = target_data.coords["time"][rho - prec_window : -1]
    else:
        warmup_length = eval_dataloader.dataset.warmup_length
        if eval_dataloader.dataset.name == "FloodEventDplDataset":
            selected_time_points = target_data.coords["time"][:]
        else:
            selected_time_points = target_data.coords["time"][warmup_length:]

    selected_data = target_data.sel(time=selected_time_points)
    if (
        eval_dataloader.dataset.name == "FloodEventDataset"
        or eval_dataloader.dataset.name == "FloodEventDplDataset"
    ):
        # FloodEventDataset has two variables: streamflow and flood_event
        selected_data = selected_data.drop_sel(variable="flood_event")
    else:
        # other datasets keep all variables
        pass
    preds_xr = target_scaler.inverse_transform(
        xr.DataArray(
            output.transpose(2, 0, 1),
            dims=selected_data.dims,
            coords=selected_data.coords,
            attrs={"units": units},
        )
    )
    obss_xr = target_scaler.inverse_transform(
        xr.DataArray(
            labels.transpose(2, 0, 1),
            dims=selected_data.dims,
            coords=selected_data.coords,
            attrs={"units": units},
        )
    )

    return preds_xr, obss_xr


class EarlyStopper(object):
    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        cumulative_delta: bool = False,
    ):
        """
        EarlyStopping handler can be used to stop the training if no improvement after a given number of events.

        Parameters
        ----------
        patience
            Number of events to wait if no improvement and then stop the training.
        min_delta
            A minimum increase in the score to qualify as an improvement,
            i.e. an increase of less than or equal to `min_delta`, will count as no improvement.
        cumulative_delta
            It True, `min_delta` defines an increase since the last `patience` reset, otherwise,
        it defines an increase after the last event. Default value is False.
        """

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score = None

    def check_loss(self, model, validation_loss, save_dir) -> bool:
        score = validation_loss
        if self.best_score is None:
            self.save_model_checkpoint(model, save_dir)
            self.best_score = score

        elif score + self.min_delta >= self.best_score:
            self.counter += 1
            print("Epochs without Model Update:", self.counter)
            if self.counter >= self.patience:
                return False
        else:
            self.save_model_checkpoint(model, save_dir)
            print("Model Update")
            self.best_score = score
            self.counter = 0
        return True

    def save_model_checkpoint(self, model, save_dir):
        # Don't save best_model.pth during training, only record the best epoch
        # The best_model.pth will be generated after training ends based on training logs
        pass


def calculate_and_record_metrics(
    obs, pred, evaluation_metrics, target_col, fill_nan, eval_log
):
    fill_nan_value = fill_nan
    inds = statistic_nd_error(obs, pred, fill_nan_value)

    for evaluation_metric in evaluation_metrics:
        eval_log[f"{evaluation_metric} of {target_col}"] = inds[
            evaluation_metric
        ].tolist()

    return eval_log


def evaluate_validation(
    validation_data_loader,
    output,
    labels,
    evaluation_cfgs,
    target_col,
):
    """
    calculate metrics for validation

    Parameters
    ----------
    output
        model output
    labels
        model target
    evaluation_cfgs
        evaluation configs
    target_col
        target columns

    Returns
    -------
    tuple
        metrics
    """
    fill_nan = evaluation_cfgs["fill_nan"]
    # if isinstance(fill_nan, list) and len(fill_nan) != len(target_col):
    #     raise ValueError("Length of fill_nan must be equal to length of target_col.")
    eval_log = {}
    batch_size = validation_data_loader.batch_size
    evaluation_metrics = evaluation_cfgs["metrics"]
    evaluator = evaluation_cfgs["evaluator"]
    if evaluation_cfgs["rolling"]:
        target_scaler = validation_data_loader.dataset.target_scaler
        target_data = target_scaler.data_target
        basin_num = len(target_data.basin)
        horizon = target_scaler.data_cfgs["forecast_length"]
        prec = target_scaler.data_cfgs["prec_window"]
        for i, col in enumerate(target_col):
            delayed_tasks = []
            for length in range(horizon):
                delayed_task = len_denormalize_delayed(
                    prec,
                    length,
                    output,
                    labels,
                    basin_num,
                    batch_size,
                    target_col,
                    validation_data_loader,
                    col,
                    evaluation_cfgs["rolling"],
                )
                delayed_tasks.append(delayed_task)
            obs_pred_results = dask.compute(*delayed_tasks)
            obs_list, pred_list = zip(*obs_pred_results)
            obs = np.concatenate(obs_list, axis=1)
            pred = np.concatenate(pred_list, axis=1)
            eval_log = calculate_and_record_metrics(
                obs,
                pred,
                evaluation_metrics,
                col,
                fill_nan[i] if isinstance(fill_nan, list) else fill_nan,
                eval_log,
            )

    else:
        if evaluator["eval_way"] == "1pace":
            pace_idx = evaluator["pace_idx"]
            output = _recover_samples_to_basin(output, validation_data_loader, pace_idx)
            if (
                validation_data_loader.dataset.name == "FloodEventDataset"
                or validation_data_loader.dataset.name == "FloodEventDplDataset"
            ):
                # FloodEventDataset has two variables: streamflow and flood_event
                labels = labels[:, :, :-1]
                if validation_data_loader.dataset.name == "FloodEventDplDataset":
                    target_col = target_col[:-1]
            else:
                pass
            labels = _recover_samples_to_basin(labels, validation_data_loader, pace_idx)
        preds_xr, obss_xr = denormalize4eval(validation_data_loader, output, labels)
        for i, col in enumerate(target_col):
            obs = obss_xr[col].to_numpy()
            pred = preds_xr[col].to_numpy()
            eval_log = calculate_and_record_metrics(
                obs,
                pred,
                evaluation_metrics,
                col,
                fill_nan[i] if isinstance(fill_nan, list) else fill_nan,
                eval_log,
            )
    return eval_log


@dask.delayed
def len_denormalize_delayed(
    prec,
    length,
    output,
    labels,
    basin_num,
    batch_size,
    target_col,
    validation_data_loader,
    col,
    rolling,
):
    # batch_size != output.shape[0]
    o = output[:, length + prec, :].reshape(basin_num, -1, len(target_col))
    l = labels[:, length + prec, :].reshape(basin_num, -1, len(target_col))
    preds_xr, obss_xr = denormalize4eval(validation_data_loader, o, l, rolling)
    obs = obss_xr[col].to_numpy()
    pred = preds_xr[col].to_numpy()
    return obs, pred


def compute_loss(
    labels: torch.Tensor, output: torch.Tensor, criterion, **kwargs
) -> float:
    """
    Function for computing the loss

    Parameters
    ----------
    labels
        The real values for the target. Shape can be variable but should follow (batch_size, time)
    output
        The output of the model
    criterion
        loss function
    validation_dataset
        Only passed when unscaling of data is needed.
    m
        defaults to 1

    Returns
    -------
    float
        the computed loss
    """
    # a = np.sum(output.cpu().detach().numpy(),axis=1)/len(output)
    # b=[]
    # for i in a:
    #     b.append([i.tolist()])
    # output = torch.tensor(b, requires_grad=True).to(torch.device("cuda"))

    if isinstance(criterion, GaussianLoss):
        if len(output[0].shape) > 2:
            g_loss = GaussianLoss(output[0][:, :, 0], output[1][:, :, 0])
        else:
            g_loss = GaussianLoss(output[0][:, 0], output[1][:, 0])
        return g_loss(labels)
    if (
        isinstance(output, torch.Tensor)
        and len(labels.shape) != len(output.shape)
        and len(labels.shape) > 1
    ):
        if labels.shape[1] == output.shape[1]:
            labels = labels.unsqueeze(2)
        else:
            labels = labels.unsqueeze(0)
    if isinstance(
        criterion, (RMSEFloodSampleLoss, PeakFocusedFloodSampleLoss)
    ):
        # labels has one more column than output, which is the flood mask
        # so we need to remove the last column of labels to get targets
        flood_mask = labels[:, :, -1:]  # Extract flood mask from last column
        targets = labels[:, :, :-1]  # Extract targets (remove last column)
        return criterion(output, targets, flood_mask)
    if isinstance(criterion, (PeakFocusedFloodEventLoss, RMSEFloodEventLoss)):
        # For PeakFocusedFloodEventLoss and RMSEFloodEventLoss, we need event_ids
        # labels has one more column than output, which is the flood mask
        flood_mask = labels[:, :, -1:]  # Extract flood mask from last column
        targets = labels[:, :, :-1]  # Extract targets (remove last column)
        # Get event_ids from kwargs
        event_ids = kwargs.get("event_ids", None)
        if event_ids is None:
            raise ValueError(
                f"{criterion.__class__.__name__} requires event_ids in kwargs. "
                "Please ensure the dataset returns event_ids and they are passed to compute_loss."
            )
        return criterion(output, targets, flood_mask, event_ids)
    if (
        isinstance(output, torch.Tensor)
        and len(labels.shape) != len(output.shape)
        and len(labels.shape) > 1
    ):
        if labels.shape[1] == output.shape[1]:
            labels = labels.unsqueeze(2)
        else:
            labels = labels.unsqueeze(0)
    assert labels.shape == output.shape
    return criterion(output, labels.float())


def torch_single_train(
    model,
    opt: optim.Optimizer,
    criterion,
    data_loader: DataLoader,
    device=None,
    scaler=None,
    **kwargs,
) -> float:
    """
    Training function for one epoch

    Parameters
    ----------
    model
        a PyTorch model inherit from nn.Module
    opt
        optimizer function from PyTorch optim.Optimizer
    criterion
        loss function
    data_loader
        object for loading data to the model
    device
        where we put the tensors and models
    scaler
        GradScaler for mixed precision training. If None, use regular precision.

    Returns
    -------
    tuple(float, int)
        loss of this epoch and number of all iterations

    Raises
    --------
    ValueError
        if nan exits, raise a ValueError
    """
    # we will set model.eval() in the validation function so here we should set model.train()
    model.train()
    n_iter_ep = 0
    running_loss = 0.0
    which_first_tensor = kwargs["which_first_tensor"]
    seq_first = which_first_tensor != "batch"
    pbar = tqdm(data_loader)

    # Determine if using AMP
    use_amp = scaler is not None
    
    for _, batch_data in enumerate(pbar):
        # Handle different data formats: (src, trg) or (src, trg, event_ids)
        if len(batch_data) == 3:
            src, trg, event_ids = batch_data
        else:
            src, trg = batch_data
            event_ids = None
        
        # Use autocast for mixed precision training
        if use_amp:
            with autocast(device_type='cuda'):
                trg, output = model_infer(seq_first, device, model, src, trg)
                # Pass event_ids to compute_loss if available
                if event_ids is not None:
                    kwargs["event_ids"] = event_ids
                loss = compute_loss(trg, output, criterion, **kwargs)
        else:
            trg, output = model_infer(seq_first, device, model, src, trg)
            # Pass event_ids to compute_loss if available
            if event_ids is not None:
                kwargs["event_ids"] = event_ids
            loss = compute_loss(trg, output, criterion, **kwargs)
        
        if loss > 100:
            print("Warning: high loss detected")
        if torch.isnan(loss):
            continue
        
        # Backward pass with or without scaler
        if use_amp:
            scaler.scale(loss).backward()  # Backpropagate with gradient scaling
            scaler.step(opt)  # Update network parameters
            scaler.update()  # Update the scale for next iteration
        else:
            loss.backward()  # Backpropagate to compute the current gradient
            opt.step()  # Update network parameters based on gradients
        
        model.zero_grad()  # clear gradient
        if loss == float("inf"):
            raise ValueError(
                "Error infinite loss detected. Try normalizing data or performing interpolation"
            )
        running_loss += loss.item()
        n_iter_ep += 1
    if n_iter_ep == 0:
        raise ValueError(
            "All batch computations of loss result in NAN. Please check the data."
        )
    total_loss = running_loss / float(n_iter_ep)
    return total_loss, n_iter_ep


def compute_validation(
    model,
    criterion,
    data_loader: DataLoader,
    device: torch.device = None,
    use_amp: bool = False,
    **kwargs,
) -> float:
    """
    Function to compute the validation loss metrics

    Parameters
    ----------
    model
        the trained model
    criterion
        torch.nn.modules.loss
    dataloader
        The data-loader of either validation or test-data
    device
        torch.device
    use_amp
        if True, use mixed precision for faster inference

    Returns
    -------
    tuple
        validation observations (numpy array), predictions (numpy array) and the loss of validation
    """
    model.eval()
    seq_first = kwargs["which_first_tensor"] != "batch"
    obs = []
    preds = []
    all_event_ids = []
    with torch.no_grad():
        for batch_data in data_loader:
            # Handle different data formats: (src, trg) or (src, trg, event_ids)
            if len(batch_data) == 3:
                src, trg, event_ids = batch_data
                if event_ids is not None:
                    # event_ids might be a list or tensor, convert to list
                    if isinstance(event_ids, (list, tuple)):
                        all_event_ids.extend(event_ids)
                    else:
                        # If it's a tensor, convert to list
                        all_event_ids.extend(event_ids.tolist() if hasattr(event_ids, 'tolist') else [event_ids])
            else:
                src, trg = batch_data
                event_ids = None
            
            # Use autocast for mixed precision inference if enabled
            if use_amp:
                with autocast(device_type='cuda'):
                    trg, output = model_infer(seq_first, device, model, src, trg)
            else:
                trg, output = model_infer(seq_first, device, model, src, trg)
            obs.append(trg)
            preds.append(output)
        # first dim is batch
        obs_final = torch.cat(obs, dim=0)
        pred_final = torch.cat(preds, dim=0)

        # Pass event_ids to compute_loss if available
        if len(all_event_ids) > 0 and len(all_event_ids) == obs_final.shape[0]:
            kwargs["event_ids"] = all_event_ids
        
        valid_loss = compute_loss(obs_final, pred_final, criterion, **kwargs)
    y_obs = obs_final.detach().cpu().numpy()
    y_pred = pred_final.detach().cpu().numpy()
    return y_obs, y_pred, valid_loss


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def _find_min_validation_loss_epoch(data):
    """
    Find the epoch with the minimum validation loss from the training log data.

    Parameters
    ----------
    data : list of dict
        A list of dictionaries containing training information, where each dictionary corresponds to an epoch.

    Returns
    -------
    tuple
        (min_epoch, min_val_loss) The epoch number with the minimum validation loss and its corresponding loss value.
        If the data is empty or no valid validation loss can be found, returns (None, None).
    """
    if not data:
        print("Input data is empty.")
        return None, None

    df = pd.DataFrame(data)

    if "epoch" not in df.columns or "validation_loss" not in df.columns:
        print("Input data is missing 'epoch' or 'validation_loss' fields.")
        return None, None

    # Define a function to extract the numerical value from `validation_loss`
    def extract_val_loss(val_loss_str):
        """
        Extract the numerical part of the validation loss from the string.

        Parameters
        ----------
        val_loss_str : str
            A string in the form "tensor(4.1230, device='cuda:2')".

        Returns
        -------
        float
            The extracted validation loss value. If extraction fails, returns positive infinity.
        """
        match = re.search(r"tensor\(([\d\.]+)", val_loss_str)
        if match:
            try:
                return float(match[1])
            except ValueError:
                return float("inf")
        return float("inf")

    # Apply function to extract the numerical part
    df["validation_loss_value"] = df["validation_loss"].apply(extract_val_loss)

    # Check if there are valid validation losses
    if df["validation_loss_value"].isnull().all():
        print("All 'validation_loss' values cannot be parsed.")
        return None, None

    # Find the minimum validation loss and the corresponding epoch
    min_idx = df["validation_loss_value"].idxmin()
    min_row = df.loc[min_idx]

    min_epoch = min_row["epoch"]
    min_val_loss = min_row["validation_loss_value"]

    return min_epoch, min_val_loss


def read_pth_from_model_loader(model_loader, model_pth_dir):
    if model_loader["load_way"] == "specified":
        test_epoch = model_loader["test_epoch"]
        weight_path = os.path.join(model_pth_dir, f"model_Ep{str(test_epoch)}.pth")
    elif model_loader["load_way"] == "best":
        weight_path = os.path.join(model_pth_dir, "best_model.pth")
        if not os.path.exists(weight_path):
            # read log file and find the best model
            log_json = read_torchhydro_log_json_file(model_pth_dir)
            if "run" not in log_json:
                raise ValueError(
                    "No best model found. You have to train the model first."
                )
            min_epoch, min_val_loss = _find_min_validation_loss_epoch(log_json["run"])
            try:
                shutil.copy2(
                    os.path.join(model_pth_dir, f"model_Ep{str(min_epoch)}.pth"),
                    os.path.join(model_pth_dir, "best_model.pth"),
                )
            except FileNotFoundError:
                # TODO: add a recursive call to find the saved best model
                raise FileNotFoundError(
                    f"The best model's weight file {os.path.join(model_pth_dir, f'model_Ep{str(min_epoch)}.pth')} does not exist."
                )
    elif model_loader["load_way"] == "latest":
        weight_path = get_lastest_file_in_a_dir(model_pth_dir)
    elif model_loader["load_way"] == "pth":
        weight_path = model_loader["pth_path"]
    else:
        raise ValueError("Invalid load_way")
    if not os.path.exists(weight_path):
        raise ValueError(f"Model file {weight_path} does not exist.")
    return weight_path


def get_lastest_logger_file_in_a_dir(dir_path):
    """Get the last logger file in a directory

    Parameters
    ----------
    dir_path : str
        the directory

    Returns
    -------
    str
        the path of the logger file
    """
    pattern = r"^\d{1,2}_[A-Za-z]+_\d{6}_\d{2}(AM|PM)\.json$"
    pth_files_lst = [
        os.path.join(dir_path, file)
        for file in os.listdir(dir_path)
        if re.match(pattern, file)
    ]
    return get_latest_file_in_a_lst(pth_files_lst)


def read_torchhydro_log_json_file(cfg_dir):
    json_files_lst = []
    json_files_ctime = []
    for file in os.listdir(cfg_dir):
        if (
            fnmatch.fnmatch(file, "*.json")
            and "_stat" not in file  # statistics json file
            and "_dict" not in file  # data cache json file
            and "_xaj_params" not in file  # XAJ parameters json file
        ):
            json_files_lst.append(os.path.join(cfg_dir, file))
            json_files_ctime.append(os.path.getctime(os.path.join(cfg_dir, file)))
    sort_idx = np.argsort(json_files_ctime)
    cfg_file = json_files_lst[sort_idx[-1]]
    return unserialize_json(cfg_file)


def get_latest_pbm_param_file(param_dir):
    """Get the latest parameter file of physics-based models in the current directory.

    Parameters
    ----------
    param_dir : str
        The directory of parameter files.

    Returns
    -------
    str
        The latest parameter file.
    """
    param_file_lst = [
        os.path.join(param_dir, f)
        for f in os.listdir(param_dir)
        if f.startswith("pb_params") and f.endswith(".csv")
    ]
    param_files = [Path(f) for f in param_file_lst]
    param_file_names_lst = [param_file.stem.split("_") for param_file in param_files]
    ctimes = [
        int(param_file_names[param_file_names.index("params") + 1])
        for param_file_names in param_file_names_lst
    ]
    return param_files[ctimes.index(max(ctimes))] if ctimes else None


def get_latest_tensorboard_event_file(log_dir):
    """Get the latest event file in the log_dir directory.

    Parameters
    ----------
    log_dir : str
        The directory where the event files are stored.

    Returns
    -------
    str
        The latest event file.
    """
    event_file_lst = [
        os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("events")
    ]
    event_files = [Path(f) for f in event_file_lst]
    event_file_names_lst = [event_file.stem.split(".") for event_file in event_files]
    ctimes = [
        int(event_file_names[event_file_names.index("tfevents") + 1])
        for event_file_names in event_file_names_lst
    ]
    return event_files[ctimes.index(max(ctimes))]


def _recover_samples_to_basin(arr_3d, valorte_data_loader, pace_idx):
    """Reorganize the 3D prediction results by basin

    Parameters
    ----------
    arr_3d : np.ndarray
        A 3D prediction array with the shape (total number of samples, number of time steps, number of features).
    valorte_data_loader: DataLoader
        The corresponding data loader used to obtain the basin-time index mapping.
    pace_idx: int
        Which time step was chosen to show.
        -1 means we chose the final value for one prediction
        positive values means we chose the results during horzion periods
        we ignore 0, because it may lead to confusion. 1 means the 1st horizon period
        TODO: when hindcast_output is not None, this part need to be modified.

    Returns
        -------
        np.ndarray
            The reorganized 3D array with the shape (number of basins, length of time, number of features).
    """
    dataset = valorte_data_loader.dataset
    basin_num = len(dataset.t_s_dict["sites_id"])
    nt = dataset.nt
    rho = dataset.rho
    warmup_len = dataset.warmup_length
    horizon = dataset.horizon
    nf = dataset.noutputvar
    if (
        valorte_data_loader.dataset.name == "FloodEventDataset"
        or valorte_data_loader.dataset.name == "FloodEventDplDataset"
    ) and nf == 0:
        # if the dataset is FloodEventDataset and no features are selected, we set nf to 1
        nf = 1

    basin_array = np.full((basin_num, nt, nf), np.nan)

    for sample_idx in range(arr_3d.shape[0]):
        # Get the basin and start time index corresponding to this sample
        # Handle both old format (3 values) and new format (4 values with event_id)
        lookup_entry = dataset.lookup_table[sample_idx]
        if len(lookup_entry) == 4:
            basin, start_time, _, _ = lookup_entry  # Ignore event_id and actual_length
        else:
            basin, start_time, _ = lookup_entry
        # Calculate the time position in the result array
        if pace_idx < 0:
            value = arr_3d[sample_idx, pace_idx, :]
            result_time_idx = start_time + rho + horizon + pace_idx
        else:
            value = arr_3d[sample_idx, pace_idx - 1, :]
            result_time_idx = start_time + rho + pace_idx - 1
        # Fill in the corresponding position
        basin_array[basin, result_time_idx, :] = value

    return basin_array
