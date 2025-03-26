from hydrodata_tl.settings.datasets_dir import DATASETS_DIR as DATASETS_DIR_TL
from hydrodata_camels.settings.datasets_dir import DATASETS_DIR as DATASETS_DIR_CAMELS
from hydromodel_dl.datasets.data_sets import (
    BaseDataset,
    DplDataset,
)

datasets_dirction_dict = {
    # Transfer learning evaluation datasets
    "CAMELS_US_ERA5Land": DATASETS_DIR_TL["CAMELS_US_ERA5Land"]["EXPORT_DIR"],
    "CAMELS_US_HydroATLAS":DATASETS_DIR_TL["CAMELS_US_HydroATLAS"]["EXPORT_DIR"],
    "CAMELS_US_HydroATLAS_ERA5Land":DATASETS_DIR_TL["CAMELS_US_HydroATLAS_ERA5Land"]["EXPORT_DIR"],
    "CAMELS_US_HydroATLAS_ERA5Land_MSWEP": DATASETS_DIR_TL["CAMELS_US_HydroATLAS_ERA5Land_MSWEP"]["EXPORT_DIR"],
    "Changdian_HydroATLAS_ERA5Land":DATASETS_DIR_TL["Changdian_HydroATLAS_ERA5Land"]["EXPORT_DIR"],
    "Changdian_HydroATLAS_ERA5Land_MSWEP": DATASETS_DIR_TL["Changdian_HydroATLAS_ERA5Land_MSWEP"]["EXPORT_DIR"],
    # Global CAMELS benchmark datasets
    "CAMELS_AUS": DATASETS_DIR_CAMELS["CAMELS_AUS"]["EXPORT_DIR"],
    "CAMELS_DE": DATASETS_DIR_CAMELS["CAMELS_DE"]["EXPORT_DIR"],
    "CAMELS_FR": DATASETS_DIR_CAMELS["CAMELS_FR"]["EXPORT_DIR"],
    "CAMELS_SE": DATASETS_DIR_CAMELS["CAMELS_SE"]["EXPORT_DIR"],
    "CAMELS_US_daymet": DATASETS_DIR_CAMELS["CAMELS_US_daymet"]["EXPORT_DIR"],
}


datasets_read_dict = {
    "LSTMDataset": BaseDataset,
    "DPLDataset": DplDataset,
} # 生成样本方法

