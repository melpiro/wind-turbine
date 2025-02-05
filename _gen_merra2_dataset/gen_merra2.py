# read nc file
import netCDF4 as nc
import pandas as pd
from datetime import datetime
import os
import numpy as np


MERRA_FOLDER = "./merra2_04-2023_10-2023/"
POWER_FILE = "./power_cet/power-04-2023_09-2023.xlsx"
# AS GEOS is UTC+0.5, you can choose to aline the power with UTC+0 or UTC+1
SHIFT_POWER = 0
BEFORE = "20230901:00"


# Geopoint 01 : 54.00, 23.1250
# Geopoint 02 : 54.00, 22.5000
# Geopoint 03 : 54.50, 23.1250
# Geopoint 04 : 54.50, 22.5000

GEOPOINT1 = (54.00, 23.1250)
GEOPOINT2 = (54.00, 22.5000)
GEOPOINT3 = (54.50, 23.1250)
GEOPOINT4 = (54.50, 22.5000)
GEOPOINTS = [GEOPOINT1, GEOPOINT2, GEOPOINT3, GEOPOINT4]

WIND_DIR = ['N','NNE','NE','NEE','E','SEE','SE','SSE','S','SSW','SW','SWW','W','NWW','NW','NNW','N']
def angle_to_direction(angle_val):
    # nan
    if np.isnan(angle_val):
        return np.nan

    idx = np.round( (angle_val % 360)/22.5,0 )
    return WIND_DIR[int(idx)]

files = os.listdir(MERRA_FOLDER)
files = [f for f in files if ".nc" in f]

deriv_df = pd.DataFrame()
file = files[0]

for file in files:

    # file = "./MERRA2_400.tavg1_2d_slv_Nx.20230401.nc4.nc4?U2M[0:23][287:289][323:326],V250[0:23][287:289][323:326],TROPT[0:23][287:289][323:326],TROPPB[0:23][287:289][323:326],T2M[0:23][287:289][323:326],TQL[0:23][287:289][323:326],T500[0:23][287:289"
    data = nc.Dataset(MERRA_FOLDER + file, mode='r')



    lons = data.variables['lon'][:]
    lats = data.variables['lat'][:]
    time = data.variables['time'][:]
    

    lats_index = []
    for i in GEOPOINTS:
        lats_index.append((abs(lats - i[0])).argmin())

    lons_index = []
    for i in GEOPOINTS:
        lons_index.append((abs(lons - i[1])).argmin())

    if (file.startswith("MERRA2_400.tavg1_2d_slv_Nx.")):
        begin_date = file.split(".")[2]
    else:
        begin_date = file.split(".")[5]
    begin_date = datetime.strptime(begin_date, '%Y%m%d')


    df = pd.DataFrame()
    for i in range(len(GEOPOINTS)):

        if (i == 0):
            time = begin_date \
                    + pd.to_timedelta(time, unit='m')

            df["Date"] = time.strftime('%Y%m%d:%H')

        DISPH = "DISPH{} [m]".format(i+1)
        PS = "PS{} [hPa]".format(i+1)
        QV10M = "QV10M{} [g/kg]".format(i+1)
        QV2M = "QV2M{} [g/kg]".format(i+1)
        SLP = "SLP{} [hPa]".format(i+1)
        T10M = "T10M{} [C]".format(i+1)
        T2M = "T2M{} [C]".format(i+1)
        WS10M = "WS10M{} [m/s]".format(i+1)
        WD10MME = "WD10MME{} [0..360]".format(i+1)
        WD10MME_DIR = "WD10MME{} [DIR]".format(i+1)
        WS2M = "WS2M{} [m/s]".format(i+1)
        WD2MME = "WD2MME{} [0..360]".format(i+1)
        WD2MME_DIR = "WD2MME{} [DIR]".format(i+1)
        WS50M = "WS50M{} [m/s]".format(i+1)
        WD50MME = "WD50MME{} [0..360]".format(i+1)
        WD50MME_DIR = "WD50MME{} [DIR]".format(i+1)


        df[DISPH] = data.variables['DISPH'][:, lats_index[i], lons_index[i]]
        df[PS] = data.variables['PS'][:, lats_index[i], lons_index[i]] / 100.0
        df[QV10M] = data.variables['QV10M'][:, lats_index[i], lons_index[i]] * 1000.0
        df[QV2M] = data.variables['QV2M'][:, lats_index[i], lons_index[i]]  * 1000.0
        df[SLP] = data.variables['SLP'][:, lats_index[i], lons_index[i]] / 100.0
        df[T10M] = data.variables['T10M'][:, lats_index[i], lons_index[i]] - 273.15
        df[T2M] = data.variables['T2M'][:, lats_index[i], lons_index[i]] - 273.15

        v2m = data.variables['V2M'][:, lats_index[i], lons_index[i]]
        v10m = data.variables['V10M'][:, lats_index[i], lons_index[i]]
        v50m = data.variables['V50M'][:, lats_index[i], lons_index[i]]
        u2m = data.variables['U2M'][:, lats_index[i], lons_index[i]]
        u10m = data.variables['U10M'][:, lats_index[i], lons_index[i]]
        u50m = data.variables['U50M'][:, lats_index[i], lons_index[i]]

        df[WS2M] = np.sqrt(v2m**2 + u2m**2)
        df[WS10M] = np.sqrt(v10m**2 + u10m**2)
        df[WS50M] = np.sqrt(v50m**2 + u50m**2)

        WD2MME_tmp = (180.0 / np.pi) * np.arctan2(-u2m , -v2m)
        df[WD2MME] = np.where(np.array(WD2MME_tmp) < 0, 360 + WD2MME_tmp, WD2MME_tmp)

        WD10MME_tmp = (180.0 / np.pi) * np.arctan2(-u10m , -v10m)
        df[WD10MME] = np.where(np.array(WD10MME_tmp) < 0, 360 + WD10MME_tmp, WD10MME_tmp)

        WD50MME_tmp = (180.0 / np.pi) * np.arctan2(-u50m , -v50m)
        df[WD50MME] = np.where(np.array(WD50MME_tmp) < 0, 360 + WD50MME_tmp, WD50MME_tmp)

        df[WD2MME_DIR] = df[WD2MME].apply(angle_to_direction)
        df[WD10MME_DIR] = df[WD10MME].apply(angle_to_direction)
        df[WD50MME_DIR] = df[WD50MME].apply(angle_to_direction)


    deriv_df = pd.concat([deriv_df, df], axis=0)

deriv_df = deriv_df.sort_values(by="Date")
deriv_df = deriv_df.reset_index(drop=True)


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

    # cat power_df with deriv_df on Date
    deriv_df = deriv_df.merge(power_df, on='Date')
else:
    print("[WARNING] No power file found")
    deriv_df["Power [kW]"] = np.nan
    
    
# apply BEFORE filter
i_loc = deriv_df[deriv_df['Date'] == BEFORE].index[0]
deriv_df = deriv_df.iloc[:i_loc]

# YYYYMMDD:HH to date
start = datetime.strptime(deriv_df['Date'][0]              , '%Y%m%d:%H').strftime('%m-%Y')
end   = datetime.strptime(deriv_df['Date'][len(deriv_df)-1], '%Y%m%d:%H').strftime('%m-%Y')
    
    
# # rename Date to Date:time [YYYMMDD:HH]
# # rename power [kW] to E53 Power [kW]
deriv_df = deriv_df.rename(columns={'Date': 'Date:time [YYYMMDD:HH]', 'Power [kW]': 'E53 Power [kW]'})

# # put power column just after date (second column, index 1)
cols = deriv_df.columns.tolist()
cols = cols[:1] + cols[-1:] + cols[1:-1]
deriv_df = deriv_df[cols]

# save to csv
FILENAME = f'merra-{start}_{end}_s{SHIFT_POWER}.csv'
deriv_df.to_csv(FILENAME, index=False, sep=';')

# cp to "../A_Dataset/EnergyPrediction/Dataset 202304_202306_cleaned_geos.csv"
os.system(f"cp ./{FILENAME} ../A_Dataset/EnergyPrediction/{FILENAME}")