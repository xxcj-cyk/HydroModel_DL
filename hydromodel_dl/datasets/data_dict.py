"""
@Author:             Yikai CHAI
@Email:              chaiyikai@mail.dlut.edu.cn
@Company:            Dalian University of Technology
@Date:               2025-07-13 00:48:36
@Last Modified by:  Yikai CHAI
@Last Modified time:2025-07-13 00:49:24
"""

from hydromodel_dl.datasets.data_sets import (
    BaseDataset,
    DplDataset,
)

datasets_dirction_dict = {
    # Transfer learning evaluation datasets
    "CAMELS_US_HydroATLAS",
    "CAMELS_US_ERA5Land",
    "CAMELS_US_MSWEP",
    "CAMELS_US_HydroATLAS_ERA5Land",
    "CAMELS_US_HydroATLAS_MSWEP_ERA5Land",
    "CAMELS_US_HydroATLAS_MSWEP_ERA5Land-har",
    "Changdian_HydroATLAS_ERA5Land",
    "Changdian_HydroATLAS_MSWEP_ERA5Land",
    # Global CAMELS benchmark datasets
    "CAMELS_AUS",
    "CAMELS_DE",
    "CAMELS_FR",
    "CAMELS_SE",
    "CAMELS_US_daymet",
    # Local Dataset of China
    "Sanxia_1D",
    "Anhui_1H",
}


datasets_read_dict = {
    "LSTMDataset": BaseDataset,
    "DPLDataset": DplDataset,
} # 生成样本方法

