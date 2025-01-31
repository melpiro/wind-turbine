import pandas as pd
from numpy_typing import np, ax
import math

def splitDataset(data, ratio:float=None, size:int=None):
    """
    Split data into train, test and validation set
    """
    if (ratio is None and size is None):
        raise ValueError("splitDataset: ratio or size must be specified")
    train = []
    test = []
    for i in range(len(data)):
        if (ratio is not None):
            split_index = int(len(data[i]) * (1 - ratio))
            train.append(data[i][:split_index])
            test .append(data[i][split_index:])
        else:
            train.append(data[i][:-size])
            test .append(data[i][-size:])

    return train, test


def read_windpower(path:str) -> pd.DataFrame:
    """
    Read a csv file and return a pandas dataframe
    """
    if (path.endswith(".csv")):
        # auto detect if sep is ; or ,
        file = open(path, "r")
        line = file.readline()
        file.close()
        occ_semi, occ_coma = line.count(";"), line.count(",")
        if (occ_semi > occ_coma): sep = ";"
        else: sep = ","
        
        df = pd.read_csv(path, sep=sep)
    else:
        df = pd.read_excel(path, engine='openpyxl')
        
    return df
        
    
DIR = {
    "N" : 0, "NNE" : 1, "NE" : 2, "NEE" : 3, "E": 4, "SEE" : 5, "SE" : 6, "SSE" : 7, "S" : 8, "SSW" : 9, "SW" : 10, "SWW" : 11, "W" : 12, "NWW" : 13, "NW" : 14, "NNW" : 15,
    np.nan: 16
}
    
def df_to_feature_array(CTX, df) -> """tuple[
        np.float32_2d[ax.time, ax.feature], 
        np.float32_2d[ax.time, ax.feature]]""":
            
    df[CTX["TARGET_FEATURE"]] = df[CTX["TARGET_FEATURE"]].apply(lambda x: x if x >= 10 else np.nan)
    df[CTX["TARGET_FEATURE"]] = df[CTX["TARGET_FEATURE"]].fillna(-1)
    y = df[CTX["TARGET_FEATURE"]].values.reshape(-1, 1)
    
    
    ## fill remaining nan with last value
    df = df.ffill()#(method="ffill")
    df = df.bfill()#(method="bfill")
    


    # create the hour column
    if ("hour" in CTX["FEATURE_MAP"]):
        date = df["Date:time [YYYMMDD:HH]"]
        date = pd.to_datetime(date, format="%Y%m%d:%H")
        df["hour"] = date.dt.hour
    
    # create the day of year column (0 to 365)
    if ("dayofyear" in CTX["FEATURE_MAP"]):
        df["dayofyear"] = date.dt.dayofyear
    if ("dayofmonth" in CTX["FEATURE_MAP"]):
        df["dayofmonth"] = date.dt.day
    if ("dayofweek" in CTX["FEATURE_MAP"]):
        df["dayofweek"] = date.dt.dayofweek
        
    df = df[CTX["USED_FEATURES"]]
    
    # convert DIR columns to int labels
    for col in df.columns:
        if "DIR" in col:
            df[col] = df[col].apply(lambda x: DIR[x])
    

    # Remove useless columns
    df = df[CTX["USED_FEATURES"]]

    
    x = df.to_numpy().astype(np.float32)
    
    return x, y





def compute_shift(start, end, dilatation):
    """
    compute needed shift to have the last timesteps at the end of the array
    """

    d = end - start
    shift = (d - (d // dilatation) * dilatation - 1) % dilatation
    return shift



