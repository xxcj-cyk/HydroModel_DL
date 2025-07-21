import logging
import re
import sys
import torch
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from torch.utils.data import Dataset
from tqdm import tqdm
from hydrodatautils.foundation.hydro_unit import streamflow_unit_conv
from hydromodel_dl.configs.config import DATE_FORMATS
from hydromodel_dl.datasets.data_scalers import ScalerHub
from hydromodel_dl.datasets.data_readers import (
    ReadDataset_BUDYKO,
    ReadDataset_CAMELS,
    ReadDataset_CHINA,
)

from hydrodatautils.foundation.hydro_data import (
    warn_if_nan,
    wrap_t_s_dict,
)

LOGGER = logging.getLogger(__name__)


def _fill_gaps_da(da: xr.DataArray, fill_nan: Optional[str] = None) -> xr.DataArray:
    """Fill gaps in a DataArray"""
    if fill_nan is None or da is None:
        return da
    assert isinstance(da, xr.DataArray), "Expect da to be DataArray (not dataset)"
    # fill gaps
    if fill_nan == "et_ssm_ignore":
        all_non_nan_idx = []
        for i in range(da.shape[0]):
            non_nan_idx_tmp = np.where(~np.isnan(da[i].values))
            all_non_nan_idx = all_non_nan_idx + non_nan_idx_tmp[0].tolist()
        # some NaN data appear in different dates in different basins
        non_nan_idx = np.unique(all_non_nan_idx).tolist()
        for i in range(da.shape[0]):
            targ_i = da[i][non_nan_idx]
            da[i][non_nan_idx] = targ_i.interpolate_na(
                dim="time", fill_value="extrapolate"
            )
    elif fill_nan == "mean":
        # fill with mean
        for var in da["variable"].values:
            var_data = da.sel(variable=var)  # select the data for the current variable
            mean_val = var_data.mean(
                dim="basin"
            )  # calculate the mean across all basins
            if warn_if_nan(mean_val):
                # when all value are NaN, mean_val will be NaN, we set mean_val to -1
                mean_val = -1
            filled_data = var_data.fillna(
                mean_val
            )  # fill NaN values with the calculated mean
            da.loc[dict(variable=var)] = (
                filled_data  # update the original dataarray with the filled data
            )
    elif fill_nan == "interpolate":
        # fill interpolation
        for i in range(da.shape[0]):
            da[i] = da[i].interpolate_na(dim="time", fill_value="extrapolate")
    else:
        raise NotImplementedError(f"fill_nan {fill_nan} not implemented")
    return da


def detect_date_format(date_str):
    for date_format in DATE_FORMATS:
        try:
            datetime.strptime(date_str, date_format)
            return date_format
        except ValueError:
            continue
    raise ValueError(f"Unknown date format: {date_str}")


class LongTermDataset(Dataset):
    """Base data set class to load and preprocess data (batch-first) using PyTorch's Dataset"""

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        """
        Parameters
        ----------
        data_cfgs
            parameters for reading source data
        is_tra_val_te
            train, vaild or test
        """
        super(LongTermDataset, self).__init__()
        self.data_cfgs = data_cfgs
        if is_tra_val_te in {"train", "valid", "test"}:
            self.is_tra_val_te = is_tra_val_te
        else:
            raise ValueError(
                "'is_tra_val_te' must be one of 'train', 'valid' or 'test' "
            )
        # load and preprocess data
        self._load_data()

    @property
    def name(self):
        return "LongTermDataset"

    @property
    def data_source(self):
        dataset_type = self.data_cfgs["source_cfgs"].get("dataset_type", "CAMELS")
        if dataset_type == "BUDYKO":
            return ReadDataset_BUDYKO(**self.data_cfgs["source_cfgs"])
        elif dataset_type == "CAMELS":
            return ReadDataset_CAMELS(**self.data_cfgs["source_cfgs"])
        if dataset_type == "CHINA":
            return ReadDataset_CHINA(**self.data_cfgs["source_cfgs"])

    @property
    def streamflow_name(self):
        return self.data_cfgs["target_cols"][0]

    @property
    def precipitation_name(self):
        return self.data_cfgs["relevant_cols"][0]

    @property
    def ngrid(self):
        """How many basins/grids in the dataset

        Returns
        -------
        int
            number of basins/grids
        """
        return len(self.basins)

    @property
    def nt(self):
        """length of longest time series in all basins

        Returns
        -------
        int
            number of longest time steps
        """
        if isinstance(self.t_s_dict["t_final_range"][0], tuple):
            trange_type_num = len(self.t_s_dict["t_final_range"])
            if trange_type_num not in [self.ngrid, 1]:
                raise ValueError(
                    "The number of time ranges should be equal to the number of basins "
                    "if you choose different time ranges for different basins"
                )
            earliest_date = None
            latest_date = None
            for start_date_str, end_date_str in self.t_s_dict["t_final_range"]:
                date_format = detect_date_format(start_date_str)

                start_date = datetime.strptime(start_date_str, date_format)
                end_date = datetime.strptime(end_date_str, date_format)

                if earliest_date is None or start_date < earliest_date:
                    earliest_date = start_date
                if latest_date is None or end_date > latest_date:
                    latest_date = end_date
            earliest_date = earliest_date.strftime(date_format)
            latest_date = latest_date.strftime(date_format)
        else:
            trange_type_num = 1
            earliest_date = self.t_s_dict["t_final_range"][0]
            latest_date = self.t_s_dict["t_final_range"][1]
        min_time_unit = self.data_cfgs["min_time_unit"]
        min_time_interval = self.data_cfgs["min_time_interval"]
        time_step = f"{min_time_interval}{min_time_unit}"
        s_date = pd.to_datetime(earliest_date)
        e_date = pd.to_datetime(latest_date)
        time_series = pd.date_range(start=s_date, end=e_date, freq=time_step)
        return len(time_series)

    @property
    def basins(self):
        """Return the basins of the dataset"""
        return self.t_s_dict["sites_id"]

    @property
    def times(self):
        """Return the times of all basins

        TODO: Although we support get different time ranges for different basins,
        we didn't implement the reading function for this case in _read_xyc method.
        Hence, it's better to choose unified time range for all basins
        """
        min_time_unit = self.data_cfgs["min_time_unit"]
        min_time_interval = self.data_cfgs["min_time_interval"]
        time_step = f"{min_time_interval}{min_time_unit}"
        if isinstance(self.t_s_dict["t_final_range"][0], tuple):
            times_ = []
            trange_type_num = len(self.t_s_dict["t_final_range"])
            if trange_type_num not in [self.ngrid, 1]:
                raise ValueError(
                    "The number of time ranges should be equal to the number of basins "
                    "if you choose different time ranges for different basins"
                )
            detect_date_format(self.t_s_dict["t_final_range"][0][0])
            for start_date_str, end_date_str in self.t_s_dict["t_final_range"]:
                s_date = pd.to_datetime(start_date_str)
                e_date = pd.to_datetime(end_date_str)
                time_series = pd.date_range(start=s_date, end=e_date, freq=time_step)
                times_.append(time_series)
        else:
            detect_date_format(self.t_s_dict["t_final_range"][0])
            trange_type_num = 1
            s_date = pd.to_datetime(self.t_s_dict["t_final_range"][0])
            e_date = pd.to_datetime(self.t_s_dict["t_final_range"][1])
            times_ = pd.date_range(start=s_date, end=e_date, freq=time_step)
        return times_

    def __len__(self):
        return self.num_samples if self.train_mode else self.ngrid

    def __getitem__(self, item: int):
        if not self.train_mode:
            x = self.x[item, :, :]
            y = self.y[item, :, :]
            if self.c is None or self.c.shape[-1] == 0:
                return torch.from_numpy(x).float(), torch.from_numpy(y).float()
            c = self.c[item, :]
            c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
            xc = np.concatenate((x, c), axis=1)
            return torch.from_numpy(xc).float(), torch.from_numpy(y).float()
        basin, idx = self.lookup_table[item]
        warmup_length = self.warmup_length
        x = self.x[basin, idx - warmup_length : idx + self.rho + self.horizon, :]
        y = self.y[basin, idx : idx + self.rho + self.horizon, :]
        if self.c is None or self.c.shape[-1] == 0:
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

    def _pre_load_data(self):
        self.train_mode = self.is_tra_val_te == "train"
        self.t_s_dict = wrap_t_s_dict(self.data_cfgs, self.is_tra_val_te)
        self.rho = self.data_cfgs["forecast_history"]
        self.warmup_length = self.data_cfgs["warmup_length"]
        self.horizon = self.data_cfgs["forecast_length"]

    def _load_data(self):
        self._pre_load_data()
        self._read_xyc()
        # normalization
        norm_x, norm_y, norm_c = self._normalize()
        self.x, self.y, self.c = self._kill_nan(norm_x, norm_y, norm_c)
        self._trans2nparr()
        self._create_lookup_table()

    def _trans2nparr(self):
        """To make __getitem__ more efficient,
        we transform x, y, c to numpy array with shape (nsample, nt, nvar)
        """
        self.x = self.x.transpose("basin", "time", "variable").to_numpy()
        self.y = self.y.transpose("basin", "time", "variable").to_numpy()
        if self.c is not None and self.c.shape[-1] > 0:
            self.c = self.c.transpose("basin", "variable").to_numpy()
            self.c_origin = self.c_origin.transpose("basin", "variable").to_numpy()
        self.x_origin = self.x_origin.transpose("basin", "time", "variable").to_numpy()
        self.y_origin = self.y_origin.transpose("basin", "time", "variable").to_numpy()

    def _normalize(self):
        scaler_hub = ScalerHub(
            self.y_origin,
            self.x_origin,
            self.c_origin,
            data_cfgs=self.data_cfgs,
            is_tra_val_te=self.is_tra_val_te,
            data_source=self.data_source,
        )
        self.target_scaler = scaler_hub.target_scaler
        return scaler_hub.x, scaler_hub.y, scaler_hub.c

    def _to_dataarray_with_unit(self, data_forcing_ds, data_output_ds, data_attr_ds):
        # trans to dataarray to better use xbatch
        if data_output_ds is not None:
            data_output = self._trans2da_and_setunits(data_output_ds)
        else:
            data_output = None
        if data_forcing_ds is not None:
            data_forcing = self._trans2da_and_setunits(data_forcing_ds)
        else:
            data_forcing = None
        if data_attr_ds is not None:
            # firstly, we should transform some str type data to float type
            data_attr = self._trans2da_and_setunits(data_attr_ds)
        else:
            data_attr = None
        return data_forcing, data_output, data_attr

    def _check_ts_xrds_unit(self, data_forcing_ds, data_output_ds):
        """Check timeseries xarray dataset unit and convert if necessary

        Parameters
        ----------
        data_forcing_ds : xr.Dataset
            the forcing data
        data_output_ds : xr.Dataset
            outputs including streamflow data
        """

        def standardize_unit(unit):
            unit = unit.lower()  # convert to lower case
            unit = re.sub(r"day", "d", unit)
            unit = re.sub(r"hour", "h", unit)
            return unit

        streamflow_unit = data_output_ds[self.streamflow_name].attrs["units"]
        prcp_unit = data_forcing_ds[self.precipitation_name].attrs["units"]

        standardized_streamflow_unit = standardize_unit(streamflow_unit)
        standardized_prcp_unit = standardize_unit(prcp_unit)
        if standardized_streamflow_unit != standardized_prcp_unit:
            streamflow_dataset = data_output_ds[[self.streamflow_name]]
            converted_streamflow_dataset = streamflow_unit_conv(
                streamflow_dataset,
                self.data_source.read_area(self.t_s_dict["sites_id"]),
                target_unit=prcp_unit,
            )
            data_output_ds[self.streamflow_name] = converted_streamflow_dataset[
                self.streamflow_name
            ]
        return data_forcing_ds, data_output_ds

    def _read_xyc(self):
        """Read x, y, c data from data source

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset, xr.Dataset]
            x, y, c data
        """
        # x
        start_date = self.t_s_dict["t_final_range"][0]
        end_date = self.t_s_dict["t_final_range"][1]
        self._read_xyc_specified_time(start_date, end_date)

    def _read_xyc_specified_time(self, start_date, end_date):
        """Read x, y, c data from data source with specified time range
        We set this function as sometimes we need adjust the time range for some specific dataset,
        such as seq2seq dataset (it needs one more period for the end of the time range)

        Parameters
        ----------
        start_date : str
            start time
        end_date : str
            end time
        """
        data_forcing_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [start_date, end_date],
            self.data_cfgs["relevant_cols"],
        )
        # y
        data_output_ds_ = self.data_source.read_ts_xrdataset(
            self.t_s_dict["sites_id"],
            [start_date, end_date],
            self.data_cfgs["target_cols"],
        )
        if isinstance(data_output_ds_, dict) or isinstance(data_forcing_ds_, dict):
            # this means the data source return a dict with key as time_unit
            # in this LongTermDataset, we only support unified time range for all basins, so we chose the first key
            # TODO: maybe this could be refactored better
            data_forcing_ds_ = data_forcing_ds_[list(data_forcing_ds_.keys())[0]]
            data_output_ds_ = data_output_ds_[list(data_output_ds_.keys())[0]]
        data_forcing_ds, data_output_ds = self._check_ts_xrds_unit(
            data_forcing_ds_, data_output_ds_
        )
        # c
        data_attr_ds = self.data_source.read_attr_xrdataset(
            self.t_s_dict["sites_id"],
            self.data_cfgs["constant_cols"],
        )
        self.x_origin, self.y_origin, self.c_origin = self._to_dataarray_with_unit(
            data_forcing_ds, data_output_ds, data_attr_ds
        )

    def _trans2da_and_setunits(self, ds):
        """Set units for dataarray transfromed from dataset"""
        result = ds.to_array(dim="variable")
        units_dict = {
            var: ds[var].attrs["units"]
            for var in ds.variables
            if "units" in ds[var].attrs
        }
        result.attrs["units"] = units_dict
        return result

    def _kill_nan(self, x, y, c):
        data_cfgs = self.data_cfgs
        y_rm_nan = data_cfgs["target_rm_nan"]
        x_rm_nan = data_cfgs["relevant_rm_nan"]
        c_rm_nan = data_cfgs["constant_rm_nan"]
        if x_rm_nan:
            # As input, we cannot have NaN values
            _fill_gaps_da(x, fill_nan="interpolate")
            warn_if_nan(x)
        if y_rm_nan:
            _fill_gaps_da(y, fill_nan="interpolate")
            warn_if_nan(y)
        if c_rm_nan:
            _fill_gaps_da(c, fill_nan="mean")
            warn_if_nan(c)
        warn_if_nan(x, nan_mode="all")
        warn_if_nan(y, nan_mode="all")
        warn_if_nan(c, nan_mode="all")
        return x, y, c

    def _create_lookup_table(self):
        lookup = []
        # list to collect basins ids of basins without a single training sample
        basin_coordinates = len(self.t_s_dict["sites_id"])
        rho = self.rho
        warmup_length = self.warmup_length
        horizon = self.horizon
        max_time_length = self.nt
        for basin in tqdm(range(basin_coordinates), file=sys.stdout, disable=False):
            if self.is_tra_val_te != "train":
                lookup.extend(
                    (basin, f)
                    for f in range(warmup_length, max_time_length - rho - horizon + 1)
                )
            else:
                # some dataloader load data with warmup period, so leave some periods for it
                # [warmup_len] -> time_start -> [rho] -> [horizon]
                nan_array = np.isnan(self.y[basin, :, :])
                lookup.extend(
                    (basin, f)
                    for f in range(warmup_length, max_time_length - rho - horizon + 1)
                    if not np.all(nan_array[f + rho : f + rho + horizon])
                )
        self.lookup_table = dict(enumerate(lookup))
        self.num_samples = len(self.lookup_table)


class FloodEventDataset(LongTermDataset):
    """Dataset class for flood event detection and prediction tasks.

    This dataset is specifically designed to handle flood event data where
    flood_event column contains binary indicators (0 for normal, non-zero for flood).
    It automatically creates a flood_mask from the flood_event data for special
    loss computation purposes.

    The dataset reads data using SelfMadeHydroDataset from hydrodatasource,
    expecting CSV files with columns like: time, rain, inflow, flood_event.
    """

    def __init__(self, data_cfgs: dict, is_tra_val_te: str):
        """Initialize FloodEventDataset

        Parameters
        ----------
        cfgs : dict
            Configuration dictionary containing data_cfgs, training_cfgs, evaluation_cfgs
        is_tra_val_te : str
            One of 'train', 'valid', or 'test'
        """
        # Find flood_event column index for later processing
        self.data_cfgs = data_cfgs
        target_cols = self.data_cfgs["target_cols"]
        self.flood_event_idx = None
        for i, col in enumerate(target_cols):
            if "flood_event" in col.lower():
                self.flood_event_idx = i
                break

        if self.flood_event_idx is None:
            raise ValueError(
                "flood_event column not found in target_cols. Please ensure flood_event is included in the target columns."
            )
        super(FloodEventDataset, self).__init__(data_cfgs, is_tra_val_te)

    @property
    def name(self):
        return "FloodEventDataset"

    @property
    def noutputvar(self):
        """How many output variables in the dataset
        Used in evaluation.
        For flood datasets, the number of output variables is 2.
        But we don't need flood_mask in evaluation.

        Returns
        -------
        int
            number of variables
        """
        return len(self.data_cfgs["target_cols"]) - 1

    def __len__(self):
        return self.num_samples

    def _create_flood_mask(self, y):
        """Create flood mask from flood_event column

        Parameters
        ----------
        y : np.ndarray
            Target data with shape [seq_len, n_targets] containing flood_event column

        Returns
        -------
        np.ndarray
            Flood mask with shape [seq_len, 1] where 1 indicates flood event, 0 indicates normal
        """
        if self.flood_event_idx >= y.shape[1]:
            raise ValueError(
                f"flood_event_idx {self.flood_event_idx} exceeds target dimensions {y.shape[1]}"
            )

        # Extract flood_event column
        flood_events = y[:, self.flood_event_idx]

        # Create binary mask: 1 for flood (non-zero), 0 for normal (zero)
        no_flood_data = min(flood_events)
        flood_mask = (flood_events != no_flood_data).astype(np.float32)

        # Reshape to maintain dimension consistency
        flood_mask = flood_mask.reshape(-1, 1)

        return flood_mask

    def _create_lookup_table(self):
        """Create lookup table based on flood events with sliding window

        This method creates samples where:
        1. For each flood event sequence:
           - In training: use sliding window to generate samples with fixed length
           - In testing: use the entire flood event sequence as one sample with its actual length
        2. Each sample covers the full sequence length without internal structure division
        """
        lookup = []

        # Calculate total sample sequence length for training/validation
        sample_seqlen = self.warmup_length + self.rho + self.horizon

        for basin_idx in tqdm(range(self.ngrid)):
            # Get flood events for this basin
            flood_events = self.y_origin[basin_idx, :, self.flood_event_idx]

            # Find flood event sequences (consecutive non-zero values)
            flood_sequences = self._find_flood_sequences(flood_events)

            for seq_start, seq_end in flood_sequences:
                # For training, use sliding window approach
                self._create_sliding_window_samples(
                    basin_idx, seq_start, seq_end, sample_seqlen, lookup
                )

        self.lookup_table = dict(enumerate(lookup))
        self.num_samples = len(self.lookup_table)

    def _find_flood_sequences(self, flood_events):
        """Find sequences of consecutive flood events

        Parameters
        ----------
        flood_events : np.ndarray
            1D array of flood event indicators

        Returns
        -------
        list
            List of tuples (start_idx, end_idx) for each flood sequence
        """
        sequences = []
        in_sequence = False
        start_idx = None

        for i, event in enumerate(flood_events):
            if event > 0 and not in_sequence:
                # Start of a new flood sequence
                in_sequence = True
                start_idx = i
            elif event == 0 and in_sequence:
                # End of current flood sequence
                in_sequence = False
                sequences.append((start_idx, i - 1))
            elif i == len(flood_events) - 1 and in_sequence:
                # End of data while in sequence
                sequences.append((start_idx, i))

        return sequences

    def _create_sliding_window_samples(
        self, basin_idx, seq_start, seq_end, sample_seqlen, lookup
    ):
        """Create samples for a flood sequence using sliding window approach with data validity check

        Parameters
        ----------
        basin_idx : int
            Index of the basin
        seq_start : int
            Start index of flood sequence
        seq_end : int
            End index of flood sequence
        sample_seqlen : int
            Maximum length of each sample (warmup_length + rho + horizon)
        lookup : list
            List to append new samples to (basin_idx, actual_start, actual_length)
        """
        # Generate sliding window samples for this flood sequence
        # Each window should include at least some flood event data

        # Calculate the range where we can place the sliding window
        # The window end should not exceed the flood sequence end
        max_window_start = min(
            seq_end - sample_seqlen + 1, self.nt - sample_seqlen
        )  # Window end should not exceed seq_end or data bounds
        min_window_start = max(
            0, seq_start - sample_seqlen + 1
        )  # Window must include at least the first flood event

        # Ensure we have a valid range
        if max_window_start < min_window_start:
            return  # Skip this flood sequence if no valid window can be created

        # Generate samples with sliding window
        for window_start in range(min_window_start, max_window_start + 1):
            window_end = window_start + sample_seqlen - 1

            # Check if the window is valid (doesn't exceed data bounds and flood sequence)
            if window_end < self.nt and window_end <= seq_end:
                # Check if this window includes at least some flood events
                window_includes_flood = (window_start <= seq_end) and (
                    window_end >= seq_start
                )

                if window_includes_flood:
                    # Find the actual valid data range within this window closest to flood
                    actual_start, actual_length = self._find_valid_data_range(
                        basin_idx, window_start, window_end, seq_start, seq_end
                    )

                    # Only add sample if we have sufficient valid data
                    if (
                        actual_length >= self.rho + self.horizon
                    ):  # At least need rho + horizon
                        lookup.append((basin_idx, actual_start, actual_length))

    def _find_valid_data_range(
        self, basin_idx, window_start, window_end, flood_start, flood_end
    ):
        """Find the continuous valid data range closest to the flood sequence

        Parameters
        ----------
        basin_idx : int
            Basin index
        window_start : int
            Start of the window to check
        window_end : int
            End of the window to check
        flood_start : int
            Start index of the flood sequence
        flood_end : int
            End index of the flood sequence

        Returns
        -------
        tuple
            (actual_start, actual_length) of the valid data range closest to flood sequence
        """
        # Get data for this basin and window
        x_window = self.x[basin_idx, window_start : window_end + 1, :]

        # Check for NaN values in both input and output
        valid_mask = ~np.isnan(x_window).any(axis=1)  # Valid if no NaN in any feature

        # Find the continuous valid sequence closest to the flood sequence
        closest_start, closest_length = self._find_closest_valid_sequence(
            valid_mask, window_start, flood_start, flood_end
        )

        if closest_length <= 0:
            return window_start, 0
        return closest_start, closest_length

    def _find_closest_valid_sequence(
        self, valid_mask, window_start, flood_start, flood_end
    ):
        """Find the continuous valid sequence closest to the flood sequence

        Parameters
        ----------
        valid_mask : np.ndarray
            Boolean array indicating valid positions within the window
        window_start : int
            Start index of the window in the original time series
        flood_start : int
            Start index of the flood sequence in the original time series
        flood_end : int
            End index of the flood sequence in the original time series

        Returns
        -------
        tuple
            (closest_start, closest_length) in original time series coordinates
        """
        if not valid_mask.any():
            return window_start, 0

        # Find all continuous valid sequences within the window
        sequences = []
        current_start = None

        for i, is_valid in enumerate(valid_mask):
            if is_valid and current_start is None:
                current_start = i
            elif not is_valid and current_start is not None:
                sequences.append((current_start, i - current_start))
                current_start = None

        # Handle case where sequence continues to the end
        if current_start is not None:
            sequences.append((current_start, len(valid_mask) - current_start))

        if not sequences:
            return window_start, 0

        # If only one sequence, return it directly
        if len(sequences) == 1:
            seq_start_rel, seq_length = sequences[0]
            seq_start_abs = window_start + seq_start_rel
            return seq_start_abs, seq_length

        # Find the sequence closest to the flood sequence
        flood_center = (flood_start + flood_end) / 2
        closest_sequence = None
        min_distance = float("inf")

        for seq_start_rel, seq_length in sequences:
            seq_start_abs = window_start + seq_start_rel
            seq_end_abs = seq_start_abs + seq_length - 1
            seq_center = (seq_start_abs + seq_end_abs) / 2

            # Calculate distance from sequence center to flood center
            distance = abs(seq_center - flood_center)

            if distance < min_distance:
                min_distance = distance
                closest_sequence = (seq_start_abs, seq_length)

        return closest_sequence or (window_start, 0)

    def __getitem__(self, item: int):
        """Get one sample from the dataset with flood mask

        Returns samples with:
        1. Variable length sequences (no padding)
        2. Flood mask for weighted loss computation
        """
        basin, start_idx, actual_length = self.lookup_table[item]
        warmup_length = self.warmup_length
        end_idx = start_idx + actual_length

        # Get input and target data for the actual valid range
        x = self.x[basin, start_idx:end_idx, :]
        y = self.y[basin, start_idx + warmup_length : end_idx, :]

        # Create flood mask from flood_event column
        flood_mask = self._create_flood_mask(y)

        # Replace the original flood_event column with the new flood_mask
        y_with_flood_mask = y.copy()
        y_with_flood_mask[:, self.flood_event_idx] = flood_mask.squeeze()

        # Handle constant features if available
        if self.c is None or self.c.shape[-1] == 0:
            return (
                torch.from_numpy(x).float(),
                torch.from_numpy(y_with_flood_mask).float(),
            )

        # Add constant features to input
        c = self.c[basin, :]
        c = np.repeat(c, x.shape[0], axis=0).reshape(c.shape[0], -1).T
        xc = np.concatenate((x, c), axis=1)

        return torch.from_numpy(xc).float(), torch.from_numpy(y_with_flood_mask).float()


class FloodEventDplDataset(FloodEventDataset):
    """Dataset class for flood event detection and prediction with differential parameter learning support.

    This dataset combines FloodEventDataset's flood event handling capabilities with
    DplDataset's data format for differential parameter learning (dPL) models.
    It handles flood event sequences and returns data in the format required for
    physical hydrological models with neural network components.
    """

    def __init__(self, cfgs: dict, is_tra_val_te: str):
        """Initialize FloodEventDplDataset

        Parameters
        ----------
        cfgs : dict
            Configuration dictionary containing data_cfgs, training_cfgs, evaluation_cfgs
        is_tra_val_te : str
            One of 'train', 'valid', or 'test'
        """
        super(FloodEventDplDataset, self).__init__(cfgs, is_tra_val_te)

        # Additional attributes for DPL functionality
        self.target_as_input = self.data_cfgs["target_as_input"]
        self.constant_only = self.data_cfgs["constant_only"]

        if self.target_as_input and (not self.train_mode):
            # if the target is used as input and train_mode is False,
            # we need to get the target data in training period to generate pbm params
            self.train_dataset = FloodEventDplDataset(cfgs, is_tra_val_te="train")

    @property
    def name(self):
        return "FloodEventDplDataset"

    def __getitem__(self, item: int):
        """Get one sample from the dataset in DPL format with flood mask

        Returns data in the format required for differential parameter learning:
        - x_train: not normalized forcing data
        - z_train: normalized data for DL model (with flood mask)
        - y_train: not normalized output data

        Parameters
        ----------
        item : int
            Index of the sample

        Returns
        -------
        tuple
            ((x_train, z_train), y_train) where:
            - x_train: torch.Tensor, not normalized forcing data
            - z_train: torch.Tensor, normalized data for DL model
            - y_train: torch.Tensor, not normalized output data with flood mask
        """
        basin, start_idx, actual_length = self.lookup_table[item]
        end_idx = start_idx + actual_length
        warmup_length = self.warmup_length
        # Get normalized data first (using parent's logic for flood mask)
        xc_norm, y_norm_with_mask = super(FloodEventDplDataset, self).__getitem__(item)

        # Get original (not normalized) data
        x_origin = self.x_origin[basin, start_idx:end_idx, :]
        y_origin = self.y_origin[basin, start_idx + warmup_length : end_idx, :]

        # Create flood mask for original y data
        flood_mask_origin = self._create_flood_mask(y_origin)
        y_origin_with_mask = y_origin.copy()
        y_origin_with_mask[:, self.flood_event_idx] = flood_mask_origin.squeeze()

        # Prepare z_train based on configuration
        if self.target_as_input:
            # y_norm and xc_norm are concatenated and used for DL model
            # the order of xc_norm and y_norm matters, please be careful!
            z_train = torch.cat((xc_norm, y_norm_with_mask), -1)
        elif self.constant_only:
            # only use attributes data for DL model
            if self.c is None or self.c.shape[-1] == 0:
                # If no constant features, use a zero tensor
                z_train = torch.zeros((actual_length, 1)).float()
            else:
                c = self.c[basin, :]
                # Repeat constants for the actual sequence length
                c_repeated = (
                    np.repeat(c, actual_length, axis=0).reshape(c.shape[0], -1).T
                )
                z_train = torch.from_numpy(c_repeated).float()
        else:
            # Use normalized input features with constants
            z_train = xc_norm.float()

        # Prepare x_train (original forcing data with constants if available)
        if self.c is None or self.c.shape[-1] == 0:
            x_train = torch.from_numpy(x_origin).float()
        else:
            c = self.c_origin[basin, :]
            c_repeated = np.repeat(c, actual_length, axis=0).reshape(c.shape[0], -1).T
            x_origin_with_c = np.concatenate((x_origin, c_repeated), axis=1)
            x_train = torch.from_numpy(x_origin_with_c).float()

        # y_train is the original output data with flood mask
        y_train = torch.from_numpy(y_origin_with_mask).float()

        return (x_train, z_train), y_train
