from hydrodataset_mini.settings.datasets_dir import DATASETS_DIR
from hydrodataset_mini.data_reader import ReadCamelsUS
from torchhydro_mini.datasets.data_sets import (
    BaseDataset,
    DplDataset,
)

datasets_dirction_dict = {
    "CAMELS_AUS": DATASETS_DIR["CAMELS_AUS"]["EXPORT_DIR"],
    "CAMELS_SE": DATASETS_DIR["CAMELS_SE"]["EXPORT_DIR"],
    "CAMELS_US": DATASETS_DIR["CAMELS_US"]["EXPORT_DIR"],
}


datasets_read_dict = {
    "CAMELSDataset": BaseDataset,
    "DplDataset": DplDataset,
} # 生成样本方法



