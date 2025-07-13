import xarray as xr
import os
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


try:
    from hydrodata_budyko.settings.datasets_dir import DATASETS_DIR as DATASETS_DIR_BUDYKO
except ImportError:
    DATASETS_DIR_BUDYKO = {}
    
try:
    from hydrodata_camels.settings.datasets_dir import DATASETS_DIR as DATASETS_DIR_CAMELS
except ImportError:
    DATASETS_DIR_CAMELS = {}
    
try:
    from hydrodata_china.settings.datasets_dir import DATASETS_DIR as DATASETS_DIR_CHINA
except ImportError:
    DATASETS_DIR_CHINA = {}


class ReadDatasets(ABC):    
    def __init__(self, dataset_type=None, source_name=None, source_path=None, time_unit=["1D"]):
        self.dataset_type = dataset_type
        self.source_name = source_name
        self.source_path = source_path
        self.time_unit = time_unit
        self.datasets_dir = self._get_datasets_dir()
    
    @abstractmethod
    def _get_datasets_dir(self) -> Dict:
        pass
    
    def read_ts_xrdataset(
        self,
        gage_id_lst: List = None,
        t_range: List = None,
        var_lst: List = None,
        **kwargs,
    ) -> Dict:
        time_units = kwargs.get("time_units", self.time_unit)
        if var_lst is None:
            return None
            
        datasets_by_time_unit = {}
        
        for time_unit in time_units:
            batch_files = self._get_batch_files(time_unit)
            selected_datasets = []
            
            for batch_file in batch_files:
                ds = xr.open_dataset(batch_file)
                all_vars = ds.data_vars
                
                if any(var not in ds.variables for var in var_lst):
                    raise ValueError(f"var_lst must all be in {all_vars}")
                    
                valid_gage_ids = [gid for gid in gage_id_lst if gid in ds["basin"].values]
                if valid_gage_ids:
                    ds_selected = ds[var_lst].sel(
                        basin=valid_gage_ids, time=slice(t_range[0], t_range[1])
                    )
                    selected_datasets.append(ds_selected)
                ds.close()
            
            if selected_datasets:
                datasets_by_time_unit[time_unit] = xr.concat(
                    selected_datasets, dim="basin"
                ).sortby("basin")
            else:
                datasets_by_time_unit[time_unit] = xr.Dataset()
                
        return datasets_by_time_unit
    
    def _get_batch_files(self, time_unit: str) -> List[str]:
        export_dir = self.datasets_dir[self.source_name]["EXPORT_DIR"]
        return [
            os.path.join(export_dir, f)
            for f in os.listdir(export_dir)
            if re.match(
                rf"^timeseries_{time_unit}_batch_[A-Za-z0-9_]+_[A-Za-z0-9_]+\.nc$",
                f,
            )
        ]
    
    def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None):
        if var_lst is None or len(var_lst) == 0:
            return None
            
        attr_file = os.path.join(
            self.datasets_dir[self.source_name]["EXPORT_DIR"], "attributes.nc"
        )
        attr = xr.open_dataset(attr_file)
        return attr[var_lst].sel(basin=gage_id_lst)
    
    def read_area(self, gage_id_lst=None):
        return self.read_attr_xrdataset(gage_id_lst, ["Area"])
    
    def read_mean_prcp(self, gage_id_lst=None, unit="mm/d"):
        pre_mm_syr = self.read_attr_xrdataset(gage_id_lst, ["p_mean"])
        da = pre_mm_syr["p_mean"]
        
        # 单位转换
        unit_conversions = {
            "mm/d": 365, "mm/day": 365,
            "mm/h": 8760, "mm/hour": 8760,
            "mm/3h": 8760/3, "mm/3hour": 8760/3,
            "mm/8d": 365/8, "mm/8day": 365/8
        }
        
        if unit not in unit_conversions:
            raise ValueError(f"unit must be one of {list(unit_conversions.keys())}")
            
        converted_data = da / unit_conversions[unit]
        converted_data.attrs["units"] = unit
        pre_mm_syr["p_mean"] = converted_data
        return pre_mm_syr


class ReadDataset_BUDYKO(ReadDatasets):  
    def _get_datasets_dir(self) -> Dict:
        return DATASETS_DIR_BUDYKO


class ReadDataset_CAMELS(ReadDatasets):    
    def _get_datasets_dir(self) -> Dict:
        return DATASETS_DIR_CAMELS


class ReadDataset_CHINA(ReadDatasets):    
    def _get_datasets_dir(self) -> Dict:
        return DATASETS_DIR_CHINA