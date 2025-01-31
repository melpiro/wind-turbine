# read real_dataset.xlsl
import pandas as pd
import os
import numpy as np
from datetime import datetime
from utils import load_power, load_global_geos, filter_features, gen_deriv_df




GEOS_FOLDER = "./geos_forecasts_utc/"
POWER_FILE = "./power_cet/power-04-2023_09-2023.xlsx"


SHIFT_POWER = 0
BEFORE = "2023901:00"

df = load_global_geos(GEOS_FOLDER, start=None, end=BEFORE, shift=SHIFT_POWER)
power = load_power(POWER_FILE, start=None, end=BEFORE, shift=SHIFT_POWER)

# target:
# Geopoint 01 : 54.00, 23.1250
# Geopoint 02 : 54.00, 22.5000
# Geopoint 03 : 54.50, 23.1250
# Geopoint 04 : 54.50, 22.5000

# available:
# Geopoint 01 : 53.75, 22.1875
# Geopoint 02 : 53.75, 22.5000
# Geopoint 03 : 53.75, 22.8125
# Geopoint 04 : 53.75, 23.1250
# Geopoint 05 : 54.00, 22.1875
# Geopoint 06 : 54.00, 22.5000
# Geopoint 07 : 54.00, 22.8125
# Geopoint 08 : 54.00, 23.1250
# Geopoint 09 : 54.25, 22.1875
# Geopoint 10 : 54.25, 22.5000
# Geopoint 11 : 54.25, 22.8125
# Geopoint 12 : 54.25, 23.1250
# Geopoint 13 : 54.50, 22.1875
# Geopoint 14 : 54.50, 22.5000
# Geopoint 15 : 54.50, 22.8125
# Geopoint 16 : 54.50, 23.1250

GEOPOINT1 = 8-1
GEOPOINT2 = 6-1
GEOPOINT3 = 16-1
GEOPOINT4 = 14-1


used_features = [ "Date", "v10m", "u50m", "ps", "u2m", "disph",
    "v50m", "t2m", "qv2m", "t10m", "v2m", "qv10m", "u10m", "slp"
]


df = filter_features(df, used_features, GEOPOINTS=[GEOPOINT1, GEOPOINT2, GEOPOINT3, GEOPOINT4])
df = gen_deriv_df(df)


df = df.merge(power, on='Date', how='inner')
    
    
# |====================================================================================================================
# | Save to csv
# |====================================================================================================================

# YYYYMMDD:HH to date
start = datetime.strptime(df['Date'][0]              , '%Y%m%d:%H').strftime('%m-%Y')
end   = datetime.strptime(df['Date'][len(df)-1], '%Y%m%d:%H').strftime('%m-%Y')
    
# # rename Date to Date:time [YYYMMDD:HH]
# # rename power [kW] to E53 Power [kW]
df = df.rename(columns={'Date': 'Date:time [YYYMMDD:HH]', 'Power [kW]': 'E53 Power [kW]'})

# # put power column just after date (second column, index 1)
cols = df.columns.tolist()
cols = cols[:1] + cols[-1:] + cols[1:-1]
df = df[cols]

# save to csv
FILENAME = f'geos-{start}_{end}_s{SHIFT_POWER}.csv'
df.to_csv(FILENAME, index=False, sep=';')

# cp to "../A_Dataset/EnergyPrediction/Dataset 202304_202306_cleaned_geos.csv"
os.system(f"cp ./{FILENAME} ../A_Dataset/EnergyPrediction/{FILENAME}")