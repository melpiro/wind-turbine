# read real_dataset.xlsl
import pandas as pd
import os
import numpy as np
from datetime import datetime
from utils import load_power, load_global_geos, list_files, load_geos, filter_features, gen_deriv_df




GEOS_FOLDER = "./geos_forecasts_utc/"
POWER_FILE = "./power_cet/power-04-2023_09-2023.xlsx"


SHIFT_POWER = 0
AFTER = "20230901:00"
BEFORE = None
MAX_HISTORY = 24


df = load_global_geos(GEOS_FOLDER, start=AFTER, end=BEFORE, shift=SHIFT_POWER)
power = load_power(POWER_FILE, start=AFTER, end=BEFORE, shift=SHIFT_POWER)

# target:
# Geopoint 01 : 54.00, 23.1250
# Geopoint 02 : 54.00, 22.5000
# Geopoint 03 : 54.50, 23.1250
# Geopoint 04 : 54.50, 22.5000

# available:
# Geopoint 00 : 53.75, 22.1875
# Geopoint 01 : 53.75, 22.5000
# Geopoint 02 : 53.75, 22.8125
# Geopoint 03 : 53.75, 23.1250
# Geopoint 04 : 54.00, 22.1875
# Geopoint 05 : 54.00, 22.5000
# Geopoint 06 : 54.00, 22.8125
# Geopoint 07 : 54.00, 23.1250
# Geopoint 08 : 54.25, 22.1875
# Geopoint 09 : 54.25, 22.5000
# Geopoint 10 : 54.25, 22.8125
# Geopoint 11 : 54.25, 23.1250
# Geopoint 12 : 54.50, 22.1875
# Geopoint 13 : 54.50, 22.5000
# Geopoint 14 : 54.50, 22.8125
# Geopoint 15 : 54.50, 23.1250

GEOPOINT1 = 7
GEOPOINT2 = 5
GEOPOINT3 = 15
GEOPOINT4 = 13


used_features = [ "Date", "v10m", "u50m", "ps", "u2m", "disph",
    "v50m", "t2m", "qv2m", "t10m", "v2m", "qv10m", "u10m", "slp"
]


df = filter_features(df, used_features, GEOPOINTS=[GEOPOINT1, GEOPOINT2, GEOPOINT3, GEOPOINT4])
df = gen_deriv_df(df)


df = df.merge(power, on='Date', how='inner')
    
start = datetime.strptime(df['Date'][0]              , '%Y%m%d:%H').strftime('%m-%Y')
end   = datetime.strptime(df['Date'][len(df)-1], '%Y%m%d:%H').strftime('%m-%Y')
    
EVAL = f'eval_geos_{start}_{end}_s{SHIFT_POWER}/'
os.makedirs(EVAL, exist_ok=True)
os.system(f'rm -rf {EVAL}*')


power_df = df[['Date', 'E53 Power [kW]']]
power_df = power_df.rename(columns={'E53 Power [kW]': 'True power'})
power_df.to_excel(EVAL + 'TruePower.xlsx', index=False)

    
geos_files = list_files(GEOS_FOLDER)
geos_files.sort()

i = 0
file = geos_files[0]
for file in geos_files:
    geos_df = load_geos(GEOS_FOLDER + file, start=AFTER, end=BEFORE, shift=SHIFT_POWER)
    if (geos_df.empty):
        continue
    geos_df = filter_features(geos_df, used_features, GEOPOINTS=[GEOPOINT1, GEOPOINT2, GEOPOINT3, GEOPOINT4])
    geos_df = gen_deriv_df(geos_df)
    
    # get the 24 last hours
    end = geos_df['Date'].iloc[0]
    i_loc = df[df['Date'] == end].index
    if (i_loc.empty):
        continue
    i_loc = i_loc[0]
    i_loc -= MAX_HISTORY
    
    if (i_loc < 0):
        continue
    
    pre_geos_df = df.iloc[i_loc:i_loc+MAX_HISTORY]
    geos_df = pd.concat([pre_geos_df, geos_df])
    cols = geos_df.columns.tolist()
    cols = cols[:1] + cols[-1:] + cols[1:-1]
    geos_df = geos_df[cols]
    
    geos_df = geos_df.rename(columns={'Date': 'Date:time [YYYMMDD:HH]', 'Power [kW]': 'E53 Power [kW]'})

    
    print(file)
    geos_df.to_excel(EVAL + "in"+str(i).zfill(3)+".xlsx", index=False)
    i += 1
    


