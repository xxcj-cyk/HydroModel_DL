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
    "xaj": {
        "param_name": [
            # Allen, R.G., L. Pereira, D. Raes, and M. Smith, 1998.
            # Crop Evapotranspiration, Food and Agriculture Organization of the United Nations,
            # Rome, Italy. FAO publication 56. ISBN 92-5-104219-5. 290p.
            "K",  # ratio of potential evapotranspiration to reference crop evaporation generally from Allen, 1998
            "B",  # The exponent of the tension water capacity curve
            "IM",  # The ratio of the impervious to the total area of the basin
            "UM",  # Tension water capacity in the upper layer
            "LM",  # Tension water capacity in the lower layer
            "DM",  # Tension water capacity in the deepest layer
            "C",  # The coefficient of deep evapotranspiration
            "SM",  # The areal mean of the free water capacity of surface soil layer
            "EX",  # The exponent of the free water capacity curve
            "KI",  # Outflow coefficients of interflow
            "KG",  # Outflow coefficients of groundwater
            "CS",  # The recession constant of channel system
            "L",  # Lag time
            "CI",  # The recession constant of the lower interflow
            "CG",  # The recession constant of groundwater storage
        ],
        "param_range": OrderedDict(
            {
                "K": [0.1, 1.0],
                "B": [0.1, 0.4],
                "IM": [0.01, 0.1],
                "UM": [0.0, 20.0],
                "LM": [60.0, 90.0],
                "DM": [60.0, 120.0],
                "C": [0.0, 0.2],
                "SM": [1, 100.0],
                "EX": [1.0, 1.5],
                "KI": [0.0, 0.7],
                "KG": [0.0, 0.7],
                "CS": [0.0, 1.0],
                "L": [1.0, 10.0],  # unit is day
                "CI": [0.0, 0.9],
                "CG": [0.98, 0.998],
            }
        ),
    },
    "xaj_mz": {
        "param_name": [
            # Allen, R.G., L. Pereira, D. Raes, and M. Smith, 1998.
            # Crop Evapotranspiration, Food and Agriculture Organization of the United Nations,
            # Rome, Italy. FAO publication 56. ISBN 92-5-104219-5. 290p.
            "K",  # ratio of potential evapotranspiration to reference crop evaporation generally from Allen, 1998
            "B",  # The exponent of the tension water capacity curve
            "IM",  # The ratio of the impervious to the total area of the basin
            "UM",  # Tension water capacity in the upper layer
            "LM",  # Tension water capacity in the lower layer
            "DM",  # Tension water capacity in the deepest layer
            "C",  # The coefficient of deep evapotranspiration
            "SM",  # The areal mean of the free water capacity of surface soil layer
            "EX",  # The exponent of the free water capacity curve
            "KI",  # Outflow coefficients of interflow
            "KG",  # Outflow coefficients of groundwater
            "A",  # parameter of mizuRoute
            "THETA",  # parameter of mizuRoute
            "CI",  # The recession constant of the lower interflow
            "CG",  # The recession constant of groundwater storage
        ],
        "param_range": OrderedDict(
            {
                # "K": [0.1, 1.0],
                # "B": [0.1, 0.4],
                # "IM": [0.01, 0.1],
                # "UM": [0.0, 20.0],
                # "LM": [60.0, 90.0],
                # "DM": [60.0, 120.0],
                # "C": [0.0, 0.2],
                # "SM": [1.0, 100.0],
                # "EX": [1.0, 1.5],
                # "KI": [0.0, 0.7],
                # "KG": [0.0, 0.7],
                # "A": [0.0, 2.9],
                # "THETA": [0.0, 6.5],
                # "CI": [0.0, 0.9],
                # "CG": [0.98, 0.998],
                "K": [0.3, 1.2],
                "B": [0.1, 0.5],
                "IM": [0.01, 0.3],
                "UM": [5.0, 30.0],
                "LM": [40.0, 120.0],
                "DM": [40.0, 150.0],
                "C": [0.05, 0.3],
                "SM": [10.0, 80.0],
                "EX": [1.0, 1.5],
                "KI": [0.1, 0.7],
                "KG": [0.1, 0.7],
                "A": [0.0, 2.9],
                "THETA": [0.0, 6.5],
                "CI": [0.5, 0.95],
                "CG": [0.95, 0.999],
            }
        ),
    },
}
