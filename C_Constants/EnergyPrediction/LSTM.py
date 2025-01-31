

LEARNING_RATE = 0.0003
EPOCHS = 32
BATCH_SIZE = 32
NB_BATCH = 64
SHUFFLE = True


HISTORY = 24 # / max 24
LOOK_AHEAD = 1
SLIDING_WINDOW=True

OUTPUT_LEN = LOOK_AHEAD
INPUT_LEN = HISTORY + LOOK_AHEAD


LAYERS = 2
UNITS = 64

DROPOUT = 0.3


# |====================================================================================================================
# | MERRA2
# |====================================================================================================================

GRIDPOINTS = [1, 2, 3, 4]
GRIDPOINT = 4
USED_FEATURES = [
    # "E53 Power [kW]",
    # "DISPH? [m]",
    "PS? [hPa]",
    "QV10M? [g/kg]",	
    "QV2M? [g/kg]",
    "SLP? [hPa]",	
    "T10M? [C]",	
    "T2M? [C]",	
    "WS10M? [m/s]",	
    "WD10MME? [0..360]",	
    # "WD10MME? [DIR]",	
    "WS2M? [m/s]",	
    "WD2MME? [0..360]",	
    # "WD2MME? [DIR]",	
    "WS50M? [m/s]",	
    "WD50MME? [0..360]",	
    # "WD50MME? [DIR]",
    "hour",
    "dayofweek",
    "dayofmonth",
    "dayofyear",
]

USED_FEATURES = [feature.replace("?", str(g)) for feature in USED_FEATURES for g in GRIDPOINTS if ("?" in feature) or g == 1]
FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(FEATURES_IN)])




