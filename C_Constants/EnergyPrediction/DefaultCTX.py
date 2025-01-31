
# use a small test Ratio because the geos dataset is small
# TEST_RATIO = 0.01
TEST_RATIO = 0.1

SLIDING_WINDOW = True


GRIDPOINTS = [1, 2, 3, 4]
GRIDPOINT = 4
USED_FEATURES = [
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
    "day"
]
USED_FEATURES = [feature.replace("?", str(g)) for feature in USED_FEATURES for g in GRIDPOINTS if ("?" in feature) or g == 1]
FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(FEATURES_IN)])

TARGET_FEATURE = "E53 Power [kW]"


MAX_BATCH_SIZE = 1024