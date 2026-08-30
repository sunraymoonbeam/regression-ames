"""Column-level knowledge about the Ames dataset.

These mappings encode facts about the data (which ratings are ordered, which
numeric codes are really categories) rather than tunable settings, so they live
here as constants instead of in config.yaml.
"""

import numpy as np

# MSSubClass is a building-class code and MoSold is a month number: both are
# nominal, and leaving them numeric would imply an ordering that does not exist.
MSSUBCLASS_MAP = {
    20: "SC20", 30: "SC30", 40: "SC40", 45: "SC45", 50: "SC50", 60: "SC60",
    70: "SC70", 75: "SC75", 80: "SC80", 85: "SC85", 90: "SC90", 120: "SC120",
    150: "SC150", 160: "SC160", 180: "SC180", 190: "SC190",
}

MONTH_MAP = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

RECODE_MAPS = {"MSSubClass": MSSUBCLASS_MAP, "MoSold": MONTH_MAP}

# Ordered categories -> integers, ALWAYS worst to best so the direction is
# consistent across every feature. Where NaN appears in the mapping it means
# "the thing is absent" (no basement, no pool, no alley access), which is
# genuinely the bottom of the scale rather than a missing observation.
ORDINAL_MAPPINGS = {
    "Alley":        {np.nan: 0, "Grvl": 1, "Pave": 2},
    "BsmtCond":     {np.nan: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtExposure": {np.nan: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {np.nan: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {np.nan: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtQual":     {np.nan: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond":    {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterQual":    {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu":  {np.nan: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "Functional":   {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8},
    "GarageCond":   {np.nan: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual":   {np.nan: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "HeatingQC":    {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual":  {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "LandSlope":    {"Sev": 1, "Mod": 2, "Gtl": 3},
    "LotShape":     {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "PavedDrive":   {"N": 0, "P": 1, "Y": 2},
    "PoolQC":       {np.nan: 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "Street":       {"Grvl": 1, "Pave": 2},
    "Utilities":    {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4},
    # Worst to best, matching every mapping above.
    "Fence":        {np.nan: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
}

# Already-numeric columns whose values carry a natural order (ratings, counts).
NUMERIC_ORDINALS = [
    "OverallQual", "OverallCond", "BsmtFullBath", "BsmtHalfBath", "FullBath",
    "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageCars",
]

ORDINAL_FEATURES = list(ORDINAL_MAPPINGS) + NUMERIC_ORDINALS

# Aggregate features: name -> component columns. Missing components are skipped,
# so an aggregate still builds if an earlier step dropped one of its inputs.
AGGREGATE_FEATURES = {
    "TotalBsmtFin": ["BsmtFinSF1", "BsmtFinSF2"],
    # Includes the FIRST floor: without it, "total square footage" is missing a storey.
    "TotalSF": ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"],
    "TotalPorch": ["OpenPorchSF", "EnclosedPorch", "ScreenPorch", "3SsnPorch", "WoodDeckSF"],
}

# Bathrooms are weighted: a half bath is half a bath, and basement baths count.
BATH_WEIGHTS = {"FullBath": 1.0, "HalfBath": 0.5, "BsmtFullBath": 1.0, "BsmtHalfBath": 0.5}

# Age at point of sale, rather than the raw calendar year. A 1998 house sold in
# 2006 and a 2002 house sold in 2010 are the same 8-year-old house.
AGE_FEATURES = {"HouseAge": ("YrSold", "YearBuilt"), "RemodAge": ("YrSold", "YearRemodAdd")}
