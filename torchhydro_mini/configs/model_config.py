from collections import OrderedDict

# Model parameters
MODEL_PARAM_DICT = {
    "hbv": {
        "param_name": [
            "BETA",  # parameter in soil routine
            "FC",  # maximum soil moisture content
            "K0",  # recession coefficient
            "K1",  # recession coefficient
            "K2",  # recession coefficient
            "LP",  # limit for potential evapotranspiration
            "PERC",  # percolation from upper to lower response box
            "UZL",  # upper zone limit
            "TT",  # temperature limit for snow/rain; distinguish rainfall from snowfall
            "CFMAX",  # degree day factor; used for melting calculation
            "CFR",  # Refreezing coefficient for water in the snowpack
            "CWH",  # Liquid water holding capacity of the snowpack
            "A",  # parameter of mizuRoute
            "THETA",  # parameter of mizuRoute
        ],
        "param_range": OrderedDict(
            {
                "BETA": [1, 6],
                "FC": [50, 1000],
                "K0": [0.05, 0.9],
                "K1": [0.01, 0.5],
                "K2": [0.001, 0.2],
                "LP": [0.2, 1],
                "PERC": [0, 10],
                "UZL": [0, 100],
                "TT": [-2.5, 2.5],  # default unit is °C
                "CFMAX": [0.5, 10],
                "CFR": [0, 0.1],
                "CWH": [0, 0.2],
                "A": [0, 2.9],
                "THETA": [0, 6.5],
            }
        ),
    },
}
