"""
Author: Wenyu Ouyang
Date: 2024-04-02 14:37:09
LastEditTime: 2024-07-10 09:26:07
LastEditors: Wenyu Ouyang
Description: A module for different data sources
FilePath: /torchhydro/torchhydro/datasets/data_sources.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import collections
import os
import numpy as np
import pandas as pd
import xarray as xr
import pint_xarray  # noqa but it is used in the code
from tqdm import tqdm

from hydroutils import hydro_time
from hydrodataset import Camels
from hydrodatasource.reader.data_source import SelfMadeHydroDataset


from torchhydro import CACHE_DIR, SETTING


data_sources_dict = {
    "camels_us": Camels,
    "selfmadehydrodataset": SelfMadeHydroDataset,
    "usgs4camels": SupData4Camels,
    "modiset4camels": ModisEt4Camels,
    "nldas4camels": Nldas4Camels,
    "smap4camels": Smap4Camels,
}
