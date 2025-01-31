# read real_dataset.xlsl
import pandas as pd
import os
import numpy as np
from datetime import datetime

GEOS_FOLDER = "./geos_forecasts_utc/"
POWER_FILE = "./power_cet/power-04-2023_09-2023.xlsx"
# AS GEOS is UTC+0.5, you can choose to aline the power with UTC+0 or UTC+1
SHIFT_POWER = 0

GEOPOINTS = 16


files = os.listdir(GEOS_FOLDER)
files = [file for file in files if "_00" in file]
files = [file for file in files if ".csv" in file]


######################################################
## Cleaning csv
######################################################   

dfs = []

# fix date anomalies
for file in files:
    # filename are YYYY-MM-DD_HH.csv
    # check that all date are valid in the file

    date = datetime.strptime(file.split(".")[0], '%Y-%m-%d_%H')

    df = pd.read_csv(GEOS_FOLDER + file)
    for i in range(len(df)):
        date_str = df["Date"][i]
        expected = date.strftime('%Y%m%d:%H')
        if date_str != expected:
            print(f"Date Anomaly found in {file} at index line {i+1}")
            print(f"Expected {expected} but got {date_str}")
            df.at[i, "Date"] = expected
            print(f"Fixed to {expected} (set parameter SAVE_FIX=True to definitly save change, always backup before in case of error)")
        date = date + pd.Timedelta(hours=1)
        

    dfs.append(df)

######################################################
## SORT CSV BY DATE 
###################################################### 
     
class sorted_csv_format:
    start: datetime
    end: datetime
    df: pd.DataFrame
    
    def __init__(self, start, end, df):
        self.start = start
        self.end = end
        self.df = df  

sorted_csv:"list[sorted_csv_format]" = []
for f in range(len(files)):
    file = files[f]
    df = dfs[f]

    date_start = df["Date"][0]
    date_end = df["Date"][len(df)-1]

    # date is in format YYYYMMDD:HH
    date_start = datetime.strptime(date_start, '%Y%m%d:%H')
    date_end = datetime.strptime(date_end, '%Y%m%d:%H')

    sorted_csv.append(sorted_csv_format(date_start, date_end, df))

# sort by date_start
sorted_csv = sorted(sorted_csv, key=lambda x: x.start)



######################################################
## KEEP ONLY THE SHORT TERM FORECAST
###################################################### 

length = []
for i in range(len(sorted_csv)-1):
    
    # crop to have continuous time
    if (sorted_csv[i].end > sorted_csv[i+1].start):
        sorted_csv[i].end = sorted_csv[i+1].start - pd.Timedelta(hours=1)
    length = (sorted_csv[i].end - sorted_csv[i].start)
    length = int(length.total_seconds() // 3600)+1
    sorted_csv[i].df = sorted_csv[i].df[:length]

    # fill gaps with nan
    if (sorted_csv[i].end + pd.Timedelta(hours=1) < sorted_csv[i+1].start):
        start = sorted_csv[i].end + pd.Timedelta(hours=1)
        end = sorted_csv[i+1].start

        pad_length = (end  - start).total_seconds() // 3600
        pad_length = int(pad_length)
        dates = pd.date_range(start=start, end=end-pd.Timedelta(hours=1), freq='h')
        dates = dates.strftime('%Y%m%d:%H')
        # add new empty rows with dates at the end of df
        pad = pd.DataFrame(index=range(pad_length), columns=sorted_csv[i].df.columns)
        pad["Date"] = dates
        
        sorted_csv[i].df = pd.concat([sorted_csv[i].df, pad])

    sorted_csv[i].start = sorted_csv[i].start.strftime('%Y%m%d:%H')
    sorted_csv[i].end = sorted_csv[i].end.strftime('%Y%m%d:%H')



######################################################
## ASSEMBLY THE FINAL DATAFRAME
###################################################### 

global_df = pd.DataFrame()


for i in range(len(sorted_csv)):
    df = sorted_csv[i].df

    columns = df.columns
    columns = columns.drop("Date")
    columns = [column.split("_")[1] for column in columns]
    columns = [int(column) for column in columns]
    min_column = min(columns)
    max_column = max(columns)

    if(min_column == 1 and max_column == 16):
        # take the collumn and rename them with the right number
        columns = list(df.columns)
        for c in range(len(columns)):
            splt = columns[c].split("_")
            if len(splt) == 2:
                n = int(splt[1])
                columns[c] = splt[0] + "_" + str(n-1)

        # rename columns
        df.columns = columns


    # add to global df
    global_df = pd.concat([global_df, df])
    global_df = global_df.reset_index(drop=True)

# # date is in format YYYYMMDD:HH
# convert to datetime
global_df['Date'] = pd.to_datetime(global_df['Date'], format='%Y%m%d:%H')

# sort by date
global_df = global_df.sort_values(by=['Date'])
global_df = global_df.reset_index(drop=True)

# convert back to string
global_df['Date'] = global_df['Date'].dt.strftime('%Y%m%d:%H')

# check if there are any date duplicates
if (global_df['Date'].duplicated().any()):
    raise Exception("Date duplicates found")


if (os.path.exists(POWER_FILE)):
    # load power.xslx
    power_df = pd.read_excel(POWER_FILE, engine='openpyxl')
    if ('Date:time [YYYMMDD:HH]' not in power_df.columns):
        power_df['Date:time [YYYMMDD:HH]'] = pd.to_datetime(power_df['Date'] + ' ' + power_df['Time'], format='%d.%m.%Y %H:%M')
        power_df = power_df.drop(columns=['Date', 'Time'])
    else:
        power_df['Date:time [YYYMMDD:HH]'] = pd.to_datetime(power_df['Date:time [YYYMMDD:HH]'], format='%Y%m%d:%H')

    # convert UTC
    power_df['Date:time [YYYMMDD:HH]'] = \
        power_df['Date:time [YYYMMDD:HH]'].dt.tz_localize('CET')\
                                          .dt.tz_convert('UTC') \
                                         + pd.Timedelta(hours=SHIFT_POWER)
    
    # add hours shift to power_df
    start = power_df['Date:time [YYYMMDD:HH]'][0]
    end = power_df['Date:time [YYYMMDD:HH]'][len(power_df)-1]

    power_df['Date'] = power_df['Date:time [YYYMMDD:HH]'].dt.strftime('%Y%m%d:%H')
    power_df = power_df.drop(columns=['Date:time [YYYMMDD:HH]'])

    # cat power_df with global_df on Date
    global_df = global_df.merge(power_df, on='Date')
else:
    print("[WARNING] No power file found")
    global_df["Power [kW]"] = np.nan
    
    
    
# YYYYMMDD:HH to date
start = datetime.strptime(global_df['Date'][0]              , '%Y%m%d:%H').strftime('%m-%Y')
end   = datetime.strptime(global_df['Date'][len(global_df)-1], '%Y%m%d:%H').strftime('%m-%Y')
    


# # rename Date to Date:time [YYYMMDD:HH]
# # rename power [kW] to E53 Power [kW]
global_df = global_df.rename(columns={'Date': 'Date:time [YYYMMDD:HH]', 'Power [kW]': 'E53 Power [kW]'})

# # put power column just after date (second column, index 1)
cols = global_df.columns.tolist()
cols = cols[:1] + cols[-1:] + cols[1:-1]
global_df = global_df[cols]

# save to csv
FILENAME = f'geos-full-{start}_{end}_s{SHIFT_POWER}.csv'
global_df.to_csv(FILENAME, index=False, sep=';')

# cp to "../A_Dataset/EnergyPrediction/Dataset 202304_202306_cleaned_geos.csv"
os.system(f"cp ./{FILENAME} ../A_Dataset/EnergyPrediction/{FILENAME}")