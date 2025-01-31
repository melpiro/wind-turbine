

LEARNING_RATE = 0.0002
EPOCHS = 80
BATCH_SIZE = 32
NB_BATCH = 64


HISTORY = 4 # / max 24
LOOK_AHEAD = 1
SLIDING_WINDOW = True # if false only predict look ahead timestamp in the future

OUTPUT_LEN = LOOK_AHEAD
INPUT_LEN = HISTORY + LOOK_AHEAD


CNN_LAYERS = 2
CNN_UNITS = 64

LSTM_LAYERS = 1
LSTM_UNITS = 64

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
    # "day"
]

# |====================================================================================================================
# | FULL GEOS
# |====================================================================================================================

# GRIDPOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# GRIDPOINT = 16
# USED_FEATURES = ['cldtmp_?', 'q500_?', 't850_?', 'v50m_?',
#  'h250_?',  'tropt_?',  't2m_?',  'troppb_?',  'u2m_?',  
#  'cldprs_?',  'u250_?',  't500_?',  'qv2m_?',  'v250_?',  
#  't250_?',  'q850_?',  'tropq_?',  'u500_?',  'u50m_?',  
#  'v850_?',  'ts_?',  'troppv_?',  'v500_?',  'disph_?',  
#  'omega500_?',  'ps_?',  'to3_?',  'h500_?',  'tox_?',  
#  'u10m_?',  'h850_?',  'qv10m_?',  'tqv_?',  'h1000_?',  
#  'v2m_?',  'v10m_?',  'tqi_?',  't10m_?',  'tql_?',  
#  'pbltop_?',  'slp_?',  'q250_?',  'u850_?',  'troppt_?', 
#  "hour", "day"]


USED_FEATURES = [feature.replace("?", str(g)) for feature in USED_FEATURES for g in GRIDPOINTS if ("?" in feature) or g == 1]
FEATURES_IN = len(USED_FEATURES)
FEATURE_MAP = dict([[USED_FEATURES[i], i] for i in range(FEATURES_IN)])
