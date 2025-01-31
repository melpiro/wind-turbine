
import xarray as xr
import datetime
import pandas as pd
import numpy as np
import os
import sys
import time
import warnings
import math
import traceback
warnings.filterwarnings("ignore")


# coordinates
MIN_LAT = 53.75
MIN_LON = 22.1875
MAX_LAT = 54.50
MAX_LON = 23.125

# retry delays in various situations
SERVER_FAIL_RETRY = 1 * 60  # min

SERVER_FAIL_MAX_RETRY = 5
TOO_MANY_RETRY = 1 * 60 * 60  # h

ALL_GATHERED_RETRY = 6 * 60 * 60  # h




ERROR_TO_TIME = {
    "TOO MANY RETRY" : TOO_MANY_RETRY,
    "ALL GATHERED" : ALL_GATHERED_RETRY,
    "SERVER FAIL" : SERVER_FAIL_RETRY
}

# url of geos5 forecast
URL='https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/fcast/tavg1_2d_slv_Nx/tavg1_2d_slv_Nx.'


lat = np.arange(MIN_LAT, MAX_LAT + 0.25, 0.25)
lon = np.arange(MIN_LON, MAX_LON + 0.3125, 0.3125)
GRID = np.array(np.meshgrid(lat, lon)).T.reshape(-1, 2)

for p in range(len(GRID)):
    print("Geopoint ", '{:02.0f}'.format(p+1), " : ",'{:.2f}'.format(GRID[p][0]),", ", '{:.4f}'.format(GRID[p][1]), sep="")



is_jupyter = 'ipykernel' in sys.modules
is_terminal = sys.stdout.isatty()


SPECTACULAR_VERBOSE = is_terminal or is_jupyter

STD_ERR = open("./err", 'w')


# get a column from a netcdf file
# retry in case of server fail
def getColumn(URL, geopoint, column_name):

    # storage for the column data
    data = None 

    try:
        # try to open the file and to get the column asked
        nc = xr.open_dataset(URL,engine='netcdf4', cache=False)
        data = nc.sel(lat=GRID[geopoint][0],lon=GRID[geopoint][1])
        data = data.variables[column_name].to_numpy()
        nc.close()

        
    except Exception as e:
        if (STD_ERR is not None):
            print("\n\n### ERROR: ###\n\n", e,"\n\n### TRACEBACK: ###\n\n", traceback.format_exc(), file=STD_ERR, flush=True)
        return None
    # if (np.random.rand() > 0.5):
    #     return None
    return data
    # return np.zeros(240)

def _round(x, n=0):
    res = int(x * 10**n + 0.5) / 10**n
    if (res == int(res)):
        return int(res)
    return res

def duration_to_string(ts):
    if (ts < 60):
        return f"{_round(ts)}s"
    elif (ts < 60 * 60):
        return f"{_round(ts/60,1)}m"
    elif (ts < 60 * 60 * 24):
        return f"{_round(ts/60/60,1)}h"
    else:
        return f"{_round(ts/60/60/24,1)}d"
    

def print_column(var_number, total_var_count, varname, max_varname_lenght, geopoint, nb_geopoint, elapsed_time, remaining_time, first = False, error = ""):
    if (SPECTACULAR_VERBOSE):
        if not(first):
            print("\r", end="")

        print(f"{var_number:2.0f}/{total_var_count:2.0f}", end=" ")
        print(f"{varname}:{' '*(max_varname_lenght-len(varname))}", end=" ")

        print(f"|{'='*geopoint}{' '*(nb_geopoint-geopoint)}|", end=" ")
        print(f"{geopoint:2.0f}/{nb_geopoint:2.0f}", end=" ")
        if error=="":
            if (geopoint == 0):
                pass
            elif (geopoint < nb_geopoint):
                print(f"Finished in :   {duration_to_string(remaining_time)} ({duration_to_string(elapsed_time+remaining_time)}){' ' * 40}", end="", flush=True)
            else:
                print(f"Downloaded in : {duration_to_string(elapsed_time)}{' ' * 40}", end="\n", flush=True)
        else:
            print(f"{error} : retry in {duration_to_string(ERROR_TO_TIME[error])}{' ' * 40}", end="", flush=True)
        

            
        
    else:
        if error=="":
            if (first):
                print(f"{var_number:2.0f}/{total_var_count:2.0f}", end=" ")
                print(f"{varname}", end="")
                print(f":{' '*(max_varname_lenght-len(varname))}|", end="", flush=True)
                # if (geopoint == 0):
                #     pass
                # else:
                #     print(f":{' '*(max_varname_lenght-len(varname))}|", end="", flush=True)
                #     print("="*geopoint, end="", flush=True)
            # =
            if (geopoint == 0):
                pass     
            else:
                if (first):
                    print(f"{'='*(geopoint-1)}", end="")
                print("=", end="", flush=True)

                if (geopoint == nb_geopoint):
                    print(f"| Downloaded in {duration_to_string(elapsed_time)}", end="\n\n", flush=True)
        else:
            print(f"{' ' * (nb_geopoint-geopoint)}|", end=" ")
            print(f"{error} : retry in {duration_to_string(ERROR_TO_TIME[error])}", end="\n", flush=True)
        
            
        





# gather a complete geos5 forecast file for given coordinates and date
def getRawDataframe(datetime):

    # date to format YYYYMMDD_HH
    date = datetime.strftime('%Y%m%d_%H')
    url = URL+date

    # get the df dates + vars names
    retry = 0
    while retry < SERVER_FAIL_MAX_RETRY:
        try:
            nc = xr.open_dataset(url,engine='netcdf4', cache=False)
            dates = nc.time.values
            vars_names = list(nc.data_vars)
            vars_names.sort()
            nc.close()
            break

        except Exception as e:
            if (STD_ERR is not None):
                print("\n\n###ERROR: ###\n\n", str(e), traceback.format_exc(), file=STD_ERR, flush=True)

            retry += 1
            if (retry == SERVER_FAIL_MAX_RETRY):
                return "TOO_MANY_RETRY"
            
            print(f"retry in {duration_to_string(SERVER_FAIL_RETRY)}", flush=True, end=" ")
            time.sleep(SERVER_FAIL_RETRY)


    max_length = max([len(x) for x in vars_names])

    # for each column, one by one, get the data
    # if the server fails, getColumn() will retry
    # automatically
    cols = []
    cols_names = []
    for c in range(len(vars_names)):
        name = vars_names[c]

        print_column(c+1, len(vars_names), name, max_length, 0, len(GRID), 0, 0, first = True)

        feature_start = time.time()
        for geopoint in range(len(GRID)):

            retry = 0
            while retry < SERVER_FAIL_MAX_RETRY:

                geopoint_start = time.time() # chronometer
                res = getColumn(url, geopoint, name)
                geopoint_elapsed = time.time() - geopoint_start 
                total_elapsed = time.time() - feature_start
                estimated_remaining = (total_elapsed / (geopoint+1)) * (len(GRID) - (geopoint+1))

                if res is not None:
                    cols.append(res)
                    cols_names.append(name+"_"+str(geopoint))
                    print_column(c+1, len(vars_names), name, max_length, geopoint+1, len(GRID), total_elapsed, estimated_remaining)
                    break
                else:
                    retry += 1
                    
                    if (retry >= SERVER_FAIL_MAX_RETRY):
                        print_column(c+1, len(vars_names), name, max_length, geopoint, len(GRID), total_elapsed, estimated_remaining, error = "TOO MANY RETRY")                    
                        return "TOO_MANY_RETRY"
                    
                    print_column(c+1, len(vars_names), name, max_length, geopoint, len(GRID), total_elapsed, estimated_remaining, error = "SERVER FAIL")                    
                    time.sleep(SERVER_FAIL_RETRY)
                    print_column(c+1, len(vars_names), name, max_length, geopoint, len(GRID), total_elapsed, estimated_remaining, error = "", first=not(SPECTACULAR_VERBOSE))                    
                

                    

    # convert each column to a pandas dataframe
    for i in range(len(cols)):
        cols[i] = pd.DataFrame(cols[i])
        cols[i].columns = [cols_names[i]]
    
    # data col
    DATE=pd.DataFrame(np.array(dates), columns=["Date"])
    # date format : YYYYMMDD:HH
    try:
        DATE["Date"] = DATE["Date"].apply(lambda x: x.strftime('%Y%m%d:%H'))
    except:
        print(DATE["Date"])
        # return getRawDataframe(datetime)

    # concat
    df=pd.concat([DATE] + cols,axis=1)

    return df


# compare today's date with the last success date
# to know which dates we need to gather today
def getRemainingDates():
    today = datetime.datetime.now()
    today = datetime.datetime(today.year, today.month, today.day) # remove hours, minutes and seconds

    # open file last_success.txt
    # if file does not exist, create it and set LAST_SUCCESS TODAY - 16 days (geos5 forecast are keeped for 16 days)
    if not os.path.isfile('last_success.txt'):
        with open('last_success.txt', 'w') as f:
            f.write(str(today - datetime.timedelta(days=16)))
            f.close()

    # read last_success.txt
    last_success = None
    with open('last_success.txt', 'r') as f:
        last_success = f.read().strip()
        f.close()

    last_success = datetime.datetime.strptime(last_success, '%Y-%m-%d %H:%M:%S')
    last_success = datetime.datetime(last_success.year, last_success.month, last_success.day, last_success.hour)

    # if today - last_success > 16 days, last_success = today - 16 days
    # if (today - last_success > datetime.timedelta(days=16)):
    #     last_success = today - datetime.timedelta(days=16)
    #     with open('last_success.txt', 'w') as f:
    #         f.write(str(last_success))
    #         f.close()
    return last_success, today


# gather until death all the geos5 data for a given area
def GEOS_scrapper():

    while True:

        last_success, today = getRemainingDates()


        # iterate over date with 1D step from last_success+1D to today
        dates = pd.date_range(
            start=last_success + datetime.timedelta(hours=6),
            end=today,
            freq='6H'
        )

        
        data = None # storage for the forecast
        success_count = 0 # count the number of files successfully gathered

        # for each date we have to gather data
        for date in dates: 

            # set hour to 06
            print("Gathering data for : ", date.strftime('%Y-%m-%d %H:%M:%S'), "(Remaining : "+str(len(dates) - success_count)+")\n", flush=True)
            # get data from NASA
            data = getRawDataframe(date)

            # if returned data is not an error string
            if (type(data) is not str):

                if not(os.path.exists("data")):
                    os.mkdir("data")
                    
                # save data to csv
                data.to_csv('data/' + date.strftime('%Y-%m-%d_%H') + '.csv', index=False)

                # update the last success date
                with open('last_success.txt', 'w') as f:
                    f.write(str(date))
                    f.close()
                # continue to the next date

            else:
                # gathering the current date throw an error
                # we stop looping over dates and retry later
                break

            print("\nSUCCESS !", flush=True)
            print("\n\n")
            success_count += 1

        ###################
        # ERROR MANAGMENT #
        ###################

        if ((len(dates) - success_count) == 0):
            # we gather all the data we had to gather for today
            # Verify in 6h if we are tomorrow
            print(f"retry in {duration_to_string(ALL_GATHERED_RETRY)}", flush=True, end=" ")
            time.sleep(ALL_GATHERED_RETRY)
            print("\n\n", flush=True)

        if (type(data) is str):

            if (data == "TOO_MANY_RETRY"):
                # server is down for longer time, retry in 30 min
                time.sleep(TOO_MANY_RETRY)
                print("\n\n", flush=True)


# startup !
GEOS_scrapper()