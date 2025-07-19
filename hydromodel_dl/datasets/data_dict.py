from hydromodel_dl.datasets.data_readers import (
    DATASETS_DIR_BUDYKO,
    DATASETS_DIR_CAMELS, 
    DATASETS_DIR_CHINA
)
from hydromodel_dl.datasets.data_sets import (
    LongTermDataset,
    FloodEventDataset,
)

datasets_dirction_dict = {}

if DATASETS_DIR_BUDYKO:
    datasets_dirction_dict.update({
        "CAMELS_US_HydroATLAS": DATASETS_DIR_BUDYKO.get("CAMELS_US_HydroATLAS", {}).get("EXPORT_DIR"),
        "CAMELS_US_ERA5Land": DATASETS_DIR_BUDYKO.get("CAMELS_US_ERA5Land", {}).get("EXPORT_DIR"),
        "CAMELS_US_MSWEP": DATASETS_DIR_BUDYKO.get("CAMELS_US_MSWEP", {}).get("EXPORT_DIR"),
        "CAMELS_US_HydroATLAS_ERA5Land": DATASETS_DIR_BUDYKO.get("CAMELS_US_HydroATLAS_ERA5Land", {}).get("EXPORT_DIR"),
        "CAMELS_US_HydroATLAS_MSWEP_ERA5Land": DATASETS_DIR_BUDYKO.get("CAMELS_US_HydroATLAS_MSWEP_ERA5Land", {}).get("EXPORT_DIR"),
        "CAMELS_US_HydroATLAS_MSWEP_ERA5Land-har": DATASETS_DIR_BUDYKO.get("CAMELS_US_HydroATLAS_MSWEP_ERA5Land-har", {}).get("EXPORT_DIR"),
        "Changdian_HydroATLAS_ERA5Land": DATASETS_DIR_BUDYKO.get("Changdian_HydroATLAS_ERA5Land", {}).get("EXPORT_DIR"),
        "Changdian_HydroATLAS_MSWEP_ERA5Land": DATASETS_DIR_BUDYKO.get("Changdian_HydroATLAS_MSWEP_ERA5Land", {}).get("EXPORT_DIR"),
    })

if DATASETS_DIR_CAMELS:
    datasets_dirction_dict.update({
        "CAMELS_AUS": DATASETS_DIR_CAMELS.get("CAMELS_AUS", {}).get("EXPORT_DIR"),
        "CAMELS_DE": DATASETS_DIR_CAMELS.get("CAMELS_DE", {}).get("EXPORT_DIR"),
        "CAMELS_FR": DATASETS_DIR_CAMELS.get("CAMELS_FR", {}).get("EXPORT_DIR"),
        "CAMELS_SE": DATASETS_DIR_CAMELS.get("CAMELS_SE", {}).get("EXPORT_DIR"),
        "CAMELS_US_daymet": DATASETS_DIR_CAMELS.get("CAMELS_US_daymet", {}).get("EXPORT_DIR"),
        "CAMELS_DE_HydroATLAS": DATASETS_DIR_CAMELS.get("CAMELS_DE_HydroATLAS", {}).get("EXPORT_DIR"),
        "CAMELS_SE_HydroATLAS": DATASETS_DIR_CAMELS.get("CAMELS_SE_HydroATLAS", {}).get("EXPORT_DIR"),
        "CAMELS_US_HydroATLAS": DATASETS_DIR_CAMELS.get("CAMELS_US_HydroATLAS", {}).get("EXPORT_DIR"),
        "CAMELS_DE_ERA5-Land": DATASETS_DIR_CAMELS.get("CAMELS_DE_ERA5-Land", {}).get("EXPORT_DIR"),
        "CAMELS_SE_ERA5-Land": DATASETS_DIR_CAMELS.get("CAMELS_SE_ERA5-Land", {}).get("EXPORT_DIR"),
    })

if DATASETS_DIR_CHINA:
    datasets_dirction_dict.update({
        "Anhui_1H": DATASETS_DIR_CHINA.get("Anhui_1H", {}).get("EXPORT_DIR"),
    })

datasets_dirction_dict = {k: v for k, v in datasets_dirction_dict.items() if v is not None}

datasets_read_dict = {
    "LongTermDataset": LongTermDataset,
    "FloodEventDataset": FloodEventDataset,
}

