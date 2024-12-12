from hydrodataset_mini.datasets_dict import DATASETS_DIR
from 

datasets_dirction_dict = {
    "CAMELS_AUS": DATASETS_DIR["CAMELS_AUS"]["DATA_DIR"],
    "CAMELS_SE": DATASETS_DIR["CAMELS_SE"]["DATA_DIR"],
    "CAMELS_US": DATASETS_DIR["CAMELS_US"]["DATA_DIR"],
}

datasets_source_dict = {
    "CAMELS_US": Camels,
    "selfmadehydrodataset": SelfMadeHydroDataset,
}
