"""
@Author:             Yikai CHAI
@Email:              chaiyikai@mail.dlut.edu.cn
@Company:            Dalian University of Technology
@Date:               2025-03-30 10:44:43
@Last Modified by:   Yikai CHAI
@Last Modified time: 2025-07-27 11:00:00
@Description:        A unified data reader for various hydrological datasets, including CAMELS, CHINA, and TL.
"""

import xarray as xr
import os
import re
import importlib
import warnings

class ReadDatasets:
    def __init__(self, dataset_name, source_name, time_unit=["1D"]):
        """
        Initialize the data reader for a specific dataset.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset, e.g., 'CAMELS', 'CHINA', or 'TL'. This is used to
            dynamically import the correct settings.
        source_name : str
            The specific source name within the dataset.
        time_unit : list, optional
            The default time unit for time-series data, by default ["1D"]
        """
        self.dataset_name = dataset_name.upper()
        self.source_name = source_name
        self.time_unit = time_unit
        self.DATASETS_DIR = self._load_dataset_settings()

    def _load_dataset_settings(self):
        """Dynamically load the DATASETS_DIR from the appropriate library."""
        try:
            if self.dataset_name == 'CAMELS':
                module_name = 'hydrodata_camels.settings.datasets_dir'
            elif self.dataset_name == 'CHINA':
                module_name = 'hydrodata_china.settings.datasets_dir'
            elif self.dataset_name == 'TL':
                module_name = 'hydrodata_tl.settings.datasets_dir'
            else:
                raise ImportError(f"Unsupported dataset: {self.dataset_name}")
            
            settings_module = importlib.import_module(module_name)
            return settings_module.DATASETS_DIR
        except ImportError:
            warnings.warn(f"Could not import settings for dataset '{self.dataset_name}'. "
                        f"Please make sure the corresponding 'hydrodata' library is installed.")
            return None

    def read_ts_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ) -> dict:
        if self.DATASETS_DIR is None:
            raise RuntimeError(f"Cannot read data because settings for '{self.dataset_name}' are not loaded.")
        time_units = kwargs.get("time_units", self.time_unit)
        if var_lst is None:
            return None
        datasets_by_time_unit = {}
        export_dir = self.DATASETS_DIR[self.source_name]["EXPORT_DIR"]
        for time_unit in time_units:
            batch_files = [
                os.path.join(export_dir, f)
                for f in os.listdir(export_dir)
                if re.match(
                    rf"^timeseries_{time_unit}_batch_[A-Za-z0-9_]+_[A-Za-z0-9_]+\.nc$",
                    f,
                )
            ]
            selected_datasets = []
            for batch_file in batch_files:
                with xr.open_dataset(batch_file) as ds:
                    if any(var not in ds.variables for var in var_lst):
                        raise ValueError(f"var_lst must all be in {list(ds.data_vars)}")
                    if valid_gage_ids := [
                        gid for gid in gage_id_lst if gid in ds["basin"].values
                    ]:
                        ds_selected = ds[var_lst].sel(
                            basin=valid_gage_ids, time=slice(t_range[0], t_range[1])
                        )
                        selected_datasets.append(ds_selected)
            if selected_datasets:
                datasets_by_time_unit[time_unit] = xr.concat(
                    selected_datasets, dim="basin"
                ).sortby("basin")
            else:
                datasets_by_time_unit[time_unit] = xr.Dataset()
        return datasets_by_time_unit

    def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None):
        if self.DATASETS_DIR is None:
            raise RuntimeError(f"Cannot read attributes because settings for '{self.dataset_name}' are not loaded.")
        if var_lst is None or len(var_lst) == 0:
            return None
        attr_file = os.path.join(self.DATASETS_DIR[self.source_name]["EXPORT_DIR"], "attributes.nc")
        with xr.open_dataset(attr_file) as attr:
            return attr[var_lst].sel(basin=gage_id_lst)

    def read_area(self, gage_id_lst=None):
        return self.read_attr_xrdataset(gage_id_lst, ["Area"])

    def read_mean_prcp(self, gage_id_lst=None, unit="mm/d"):
        pre_mm_syr = self.read_attr_xrdataset(gage_id_lst, ["p_mean"])
        da = pre_mm_syr["p_mean"]
        if unit in ["mm/d", "mm/day"]:
            converted_data = da / 365
        elif unit in ["mm/h", "mm/hour"]:
            converted_data = da / 8760
        elif unit in ["mm/3h", "mm/3hour"]:
            converted_data = da / (8760 / 3)
        elif unit in ["mm/8d", "mm/8day"]:
            converted_data = da / (365 / 8)
        else:
            raise ValueError("Unsupported unit for mean precipitation.")
        converted_data.attrs["units"] = unit
        pre_mm_syr["p_mean"] = converted_data
        return pre_mm_syr
