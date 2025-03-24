from hydrodata_tl.settings.datasets_dir import DATASETS_DIR
from hydromodel_dl.datasets.data_sets import (
    BaseDataset,
    DplDataset,
)

datasets_dirction_dict = {
    "CAMELS_AUS": DATASETS_DIR["CAMELS_AUS"]["EXPORT_DIR"],
    "CAMELS_DE": DATASETS_DIR["CAMELS_DE"]["EXPORT_DIR"],
    "CAMELS_FR": DATASETS_DIR["CAMELS_FR"]["EXPORT_DIR"],
    "CAMELS_SE": DATASETS_DIR["CAMELS_SE"]["EXPORT_DIR"],
    "CAMELS_US_daymet": DATASETS_DIR["CAMELS_US_daymet"]["EXPORT_DIR"],
    "CAMELS_US_ouyang": DATASETS_DIR["CAMELS_US_ouyang"]["EXPORT_DIR"],
}


datasets_read_dict = {
    "LSTMDataset": BaseDataset,
    "DPLDataset": DplDataset,
} # 生成样本方法

