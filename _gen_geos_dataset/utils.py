
import pandas as pd
import os
from datetime import datetime
import numpy as np


# |====================================================================================================================
# | Load power data
# |====================================================================================================================


def load_power(file:str, start:str, end:str, shift:int=0) -> pd.DataFrame:
    
    if (os.path.exists(file)):


        # load power.xslx
        power_df = pd.read_excel(file, engine='openpyxl')
        if ('Date:time [YYYMMDD:HH]' not in power_df.columns):
            power_df['Date'] = pd.to_datetime(power_df['Date'] + ' ' + power_df['Time'], format='%d.%m.%Y %H:%M')
            power_df = power_df.drop(columns=['Time'])
        else:
            power_df["Date"] = pd.to_datetime(power_df['Date:time [YYYMMDD:HH]'], format='%Y%m%d:%H')
            power_df = power_df.drop(columns=['Date:time [YYYMMDD:HH]'])

        # convert UTC+shift
        power_df['Date'] = \
            power_df['Date'].dt.tz_localize('CET')\
                                            .dt.tz_convert('UTC') \
                                            .dt.tz_localize(None) \
                                            + pd.Timedelta(hours=shift)
                                            
        
        # filter by start and end
        start, end = process_date_limit(start, end)
        
        start = start.replace(tzinfo=None)
        end = end.replace(tzinfo=None)

        power_df = power_df[(power_df['Date'] >= start) & (power_df['Date'] < end)]
        power_df["Date"] = power_df["Date"].dt.strftime('%Y%m%d:%H')
        
        return power_df
    
    return pd.DataFrame([])

# |====================================================================================================================
# | Load geos data
# |====================================================================================================================

def load_global_geos(folder, start=None, end=None, shift:int=0):
    
    start, end = process_date_limit(start, end)
    files = list_files(folder)

    # load files
    dfs:"list[pd.DataFrame]" = []
    for file in files:
        df = __load_geos__(folder+file, start, end, shift)
        if (len(df) > 0):
            dfs.append(df)

    # Sort and crop data
    dfs = sorted(dfs, key=lambda x: x['Date'][0])

# |--------------------------------------------------------------------------------------------------------------------
# | Crop and fill gaps
# |--------------------------------------------------------------------------------------------------------------------

    for i in range(len(dfs)-1):
        # crop to have continuous time
        if (dfs[i]["Date"][len(dfs[i])-1] > dfs[i+1]["Date"][0]):   
            dfs[i] = dfs[i][dfs[i]["Date"] < dfs[i+1]["Date"][0]]
       
        # fill gaps with nan
        if (dfs[i]["Date"][len(dfs[i])-1] + pd.Timedelta(hours=1) < dfs[i+1]["Date"][0]):
            start = dfs[i]["Date"][len(dfs[i])-1] + pd.Timedelta(hours=1)
            end = dfs[i+1]["Date"][0]

            pad_length = (end  - start).total_seconds() // 3600
            pad_length = int(pad_length)
            dates = pd.date_range(start=start, end=end-pd.Timedelta(hours=1), freq='h')
            dates = dates.strftime('%Y%m%d:%H')
            # add new empty rows with dates at the end of df
            pad = pd.DataFrame(index=range(pad_length), columns=dfs[i].columns)
            pad["Date"] = dates
            
            dfs[i] = pd.concat([dfs[i], pad])


# |--------------------------------------------------------------------------------------------------------------------
# | ASSEMBLY THE FINAL DATAFRAME
# |--------------------------------------------------------------------------------------------------------------------

    global_df = pd.DataFrame()

    for i in range(len(dfs)):
        df = dfs[i]

        # add to global df
        global_df = pd.concat([global_df, df])
        global_df = global_df.reset_index(drop=True)
        
    # remove Date >= end
    global_df = global_df[global_df['Date'] < end]

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
    
    return global_df

def process_date_limit(start:str, end:str) -> "tuple[datetime, datetime]":
    if start is not None:
        start = datetime.strptime(start, '%Y%m%d:%H')
    else:
        start = datetime.strptime("1800-01-01", '%Y-%m-%d')
    
    if end is not None:
        end = datetime.strptime(end, '%Y%m%d:%H')
    else:
        end = datetime.strptime("9999-12-31", '%Y-%m-%d')
    
    return (start, end)
    

def list_files(folder:str) -> "list[str]":
    files = os.listdir(folder)
    files = [file for file in files if "_00" in file]
    return [file for file in files if ".csv" in file]

def fix_geopoint_issue(df:pd.DataFrame):
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
    
    return df

def load_geos(file:str, start:str, end:str, shift:int=0) -> pd.DataFrame:
    start, end = process_date_limit(start, end)
    df = __load_geos__(file, start, end, shift)
    if (df.empty):
        return df
    df["Date"] = df["Date"].dt.strftime('%Y%m%d:%H')
    return df

def __load_geos__(file, start:datetime, end:datetime, shift:int=0) -> pd.DataFrame:
    
    date = datetime.strptime(file.split("/")[-1].split("_")[0], '%Y-%m-%d')
    if (date >= end):
        return pd.DataFrame([])
    if (date < start - pd.Timedelta(days=10)):
        return pd.DataFrame([])
    
    df = pd.read_csv(file)
    df = fix_date_anomalies(df, file)
    
    df['Date'] = df["Date"] + pd.Timedelta(hours=shift)
    
    if (df['Date'][0] < start):
        return pd.DataFrame([])
    if (df['Date'][len(df)-1] >= end):
        return pd.DataFrame([])
    
    df = fix_geopoint_issue(df)
    
    return df

def fix_date_anomalies(df, file:str):
    try:
        date = datetime.strptime(df["Date"][0], '%Y%m%d:%H')
    except TypeError as e:
        # the first date is not in the right format
        print(f"Date Anomaly found in {file} at index line 1")
        raise e
    
    for i in range(len(df)):
        date_str = df["Date"][i]
        expected = date.strftime('%Y%m%d:%H')
        if date_str != expected:
            print(f"Date Anomaly found in {file} at index line {i+1}")
            print(f"Expected {expected} but got {date_str}")
            df.at[i, "Date"] = expected
            print(f"Fixed to {expected}\n")
        date = date + pd.Timedelta(hours=1)
        
    df["Date"] = pd.to_datetime(df["Date"], format='%Y%m%d:%H')
    return df
            
            
def filter_features(df:pd.DataFrame, used_features, GEOPOINTS=None):
    
    # only save the 4 geopoints
    columns = []
    COLUMNS = list(df.columns)
    for c in COLUMNS:
        split = c.split("_")
        if len(split) == 2:
            if (split[0] not in used_features):
                continue

            geopoint = int(split[1])
            
            if GEOPOINTS is not None:
                if geopoint not in GEOPOINTS:
                    continue
                
            columns.append(c)
        else:
            columns.append(c)

    df = df[columns]
    
    # rename columns with the right geopoint value
    columns = list(df.columns)
    for c in range(len(columns)):
        splt = columns[c].split("_")
        if len(splt) == 2:
            n = int(splt[1])
            g = GEOPOINTS.index(n) + 1
            columns[c] = splt[0] + "_" + str(g)
            
    df.columns = columns
    
    return df




def isDigit(c):
    return c >= '0' and c <= '9'

WIND_DIR = ['N','NNE','NE','NEE','E','SEE','SE','SSE','S','SSW','SW','SWW','W','NWW','NW','NNW','N']
def angle_to_direction(angle_val):
    # nan
    if np.isnan(angle_val):
        return np.nan

    idx = np.round( (angle_val % 360)/22.5,0 )
    return WIND_DIR[int(idx)]


def gen_deriv_df(df):
        

    columns =["Date"]
    for i in range(4):
        columns.append("DISPH{} [m]".format(i+1))
        columns.append("PS{} [hPa]".format(i+1))
        columns.append("QV10M{} [g/kg]".format(i+1))
        columns.append("QV2M{} [g/kg]".format(i+1))
        columns.append("SLP{} [hPa]".format(i+1))
        columns.append("T10M{} [C]".format(i+1))
        columns.append("T2M{} [C]".format(i+1))
        columns.append("WS10M{} [m/s]".format(i+1))
        columns.append("WD10MME{} [0..360]".format(i+1))
        columns.append("WD10MME{} [DIR]".format(i+1))
        columns.append("WS2M{} [m/s]".format(i+1))
        columns.append("WD2MME{} [0..360]".format(i+1))
        columns.append("WD2MME{} [DIR]".format(i+1))
        columns.append("WS50M{} [m/s]".format(i+1))
        columns.append("WD50MME{} [0..360]".format(i+1))
        columns.append("WD50MME{} [DIR]".format(i+1))
        
    

    deriv_df = pd.DataFrame(columns=columns)
    deriv_df["Date"] = df["Date"]

    for i in range(4):
        
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

        v10m = "v10m_{}".format(i+1)
        u50m = "u50m_{}".format(i+1)
        ps = "ps_{}".format(i+1)
        u2m = "u2m_{}".format(i+1)
        disph = "disph_{}".format(i+1)
        v50m = "v50m_{}".format(i+1)
        t2m = "t2m_{}".format(i+1)
        qv2m = "qv2m_{}".format(i+1)
        t10m = "t10m_{}".format(i+1)
        v2m = "v2m_{}".format(i+1)
        qv10m = "qv10m_{}".format(i+1)
        u10m = "u10m_{}".format(i+1)
        slp = "slp_{}".format(i+1)
        

        deriv_df[DISPH] = df[disph]
        deriv_df[PS] = df[ps] / 100.0
        deriv_df[QV10M] = df[qv10m] * 1000.0
        deriv_df[QV2M] = df[qv2m] * 1000.0
        deriv_df[SLP] = df[slp] / 100.0
        deriv_df[T10M] = df[t10m] - 273.15
        deriv_df[T2M] = df[t2m] - 273.15



        deriv_df[WS2M] = np.power(df[u2m],2)  + np.power(df[v2m],2)
        deriv_df[WS2M] = np.power(deriv_df[WS2M],0.5)

        deriv_df[WS10M] = np.power(df[v10m],2)  + np.power(df[u10m],2)
        deriv_df[WS10M] = np.power(deriv_df[WS10M],0.5)

        deriv_df[WS50M] = np.power(df[u50m],2)  + np.power(df[v50m],2)
        deriv_df[WS50M] = np.power(deriv_df[WS50M],0.5)



        WD2MME_tmp = (180.0 / np.pi) * np.arctan2( -1*df[u2m] , -1*df[v2m] )
        deriv_df[WD2MME] = WD2MME_tmp.apply( lambda x: x if x > 0 else 360 + x )

        WD10MME_tmp =  (180.0 / np.pi) * np.arctan2((-1*df[u10m]) , (-1*df[v10m]) )
        deriv_df[WD10MME] = WD10MME_tmp.apply( lambda x: x if x > 0 else 360 + x )

        WD50MME_tmp = (180.0 / np.pi) * np.arctan2(  -1*df[u50m] , -1*df[v50m] )
        deriv_df[WD50MME] = WD50MME_tmp.apply( lambda x: x if x > 0 else 360 + x )

        deriv_df[WD2MME_DIR] = deriv_df[WD2MME].apply(angle_to_direction)
        deriv_df[WD10MME_DIR] = deriv_df[WD10MME].apply(angle_to_direction)
        deriv_df[WD50MME_DIR] = deriv_df[WD50MME].apply(angle_to_direction)

    return deriv_df
