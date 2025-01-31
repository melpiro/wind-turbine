





LEARNING_RATE = 0.002
N_ESTIMATORS = 12000

EPOCHS = 1 # do not change, there is no epochs in catboost !!!!!
NB_BATCH =  1 
BATCH_SIZE = 10000 # size of the dataset



HISTORY = 4 # / max 24
LOOK_AHEAD = 1 # catboost can only predict one step ahead !!!!
SLIDING_WINDOW = True # if false only predict look ahead timestamp in the future

OUTPUT_LEN = LOOK_AHEAD
INPUT_LEN = HISTORY + LOOK_AHEAD


SHUFFLE = True

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
    # "hour",
    # "day"
]

USED_FEATURES = [feature.replace("?", str(g)) for feature in USED_FEATURES for g in GRIDPOINTS if ("?" in feature) or g == 1]
FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(FEATURES_IN)])
