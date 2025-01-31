
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
SERVER_FAIL_RETRY = 2 * 60  # min
SERVER_FAIL_BREAK = 30 * 60  # min
# NOT_AVAILABLE_RETRY = 2 * 60 * 60  # h
NOT_AVAILABLE_RETRY = 1 * 60  # h
ALL_GATHERED_RETRY = 6 * 60 * 60  # h

# max retry allowed for one column
SERVER_FAIL_MAX_RETRY = 5

# url of geos5 forecast
URL='https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/fcast/tavg1_2d_slv_Nx/tavg1_2d_slv_Nx.'



lat = np.arange(MIN_LAT, MAX_LAT + 0.25, 0.25)
lon = np.arange(MIN_LON, MAX_LON + 0.3125, 0.3125)
GRID = np.array(np.meshgrid(lat, lon)).T.reshape(-1, 2)

# for p in range(len(GRID)):
#     print("Geopoint ", '{:02.0f}'.format(p+1), " : ",'{:.2f}'.format(GRID[p][0]),", ", '{:.4f}'.format(GRID[p][1]), sep="")



is_jupyter = 'ipykernel' in sys.modules
is_terminal = sys.stdout.isatty()


SPECTACULAR_VERBOSE = is_terminal or is_jupyter

STD_ERR = open("./err", 'w')


# get a column from a netcdf file
# retry in case of server fail
def getColumn(URL, geopoint, column_name):

    # storage for the column data
    data = None 

    # compute grid coordinates from MIN to MAX
    # lat delta = 0.25
    # lon delta = 0.3125
    

    # while we didn't succeed to get the data we retry
    while data is None:

        ts = time.time() # init chronometer

        try:
            # try to open the file and to get the column asked
            nc = xr.open_dataset(URL,engine='netcdf4', cache=False)
            data = nc.sel(lat=GRID[geopoint][0],lon=GRID[geopoint][1])
            data = data.variables[column_name].to_numpy()
            nc.close()

            
        except Exception as e:
            if (STD_ERR is not None):
                print("\n\n### ERROR: ###\n\n", e,"\n\n### TRACEBACK: ###\n\n", traceback.format_exc(), file=STD_ERR, flush=True)

            # if the error is -70, it means that the URL is invalid
            # (because the data is not available yet)
            if ("[Errno -70] NetCDF: DAP server error:" in str(e)):
                return "NOT AVAILABLE"

            else:
                return "SERVER FAIL"
                




    return data

def print_error(msg):
    print(msg, end=" : ")
    if (msg == "SERVER FAIL"):
        print(f"retry in {duration_to_string(SERVER_FAIL_RETRY)}", end=" ")
    elif (msg == "NOT AVAILABLE"):
        print(f"retry in {duration_to_string(NOT_AVAILABLE_RETRY)}", end=" ")
    elif (msg == "TOO MANY RETRY"):
        print(f"retry in {duration_to_string(SERVER_FAIL_BREAK)}", end=" ")

    if(SPECTACULAR_VERBOSE):
        print(" "*50, end="", flush=True)
    else:
        print(flush=True)

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


# gather a complete geos5 forecast file for given coordinates and date
def getRawDataframe(datetime):

    # date to format YYYYMMDD_HH
    date = datetime.strftime('%Y%m%d_%H')

    url = URL+date
    #Panda's dataframe to store the retrieved data
    
    try:
        # try to open the file
        # if it fails, the data is not available yet (NOT AVAILABLE error)
        print(url)
        nc = xr.open_dataset(url,engine='netcdf4', cache=False)
        print("OK open")
        
        # Get all the dates
        dates = nc.time.values
        vars_names = list(nc.data_vars)
        vars_names.sort()

        nc.close()


        max_length = max([len(x) for x in vars_names])

        # for each column, one by one, get the data
        # if the server fails, getColumn() will retry
        # automatically
        cols = []
        cols_names = []
        if not(SPECTACULAR_VERBOSE):
            print( "              "," "*max_length, "   |", "-"*len(GRID), "|",sep="", end="\n", flush=True)

        for c in range(len(vars_names)):
            name = vars_names[c]

            # isatty is True if the output is a terminal
            if SPECTACULAR_VERBOSE:
                print("\r", end="")
                print(f"{c+1:2.0f}/{len(vars_names):2.0f} {name}_{1} |", end="")
                for gp in range(0, len(GRID)):
                    print(" ", end="")
                print("|", end="")
                print(f" {0}/{len(GRID)}", end="", flush=True)
            else:
                print(f"{c+1:2.0f}/{len(vars_names):2.0f} Getting {' ' * (max_length-len(name))}{name} : |",sep="", end="", flush=True)

            feature_start = time.time()
            for geopoint in range(len(GRID)):

                retry = 0
                while retry < SERVER_FAIL_MAX_RETRY:

                    
                    geopoint_start = time.time() # chronometer
                    res = getColumn(url, geopoint, name)
                    geopoint_elapsed = time.time() - geopoint_start 
                    total_elapsed = time.time() - feature_start
                    estimated_remaining = (total_elapsed / (geopoint+1)) * (len(GRID) - (geopoint+1))
                    
                    # if getColumn() didn't succeed
                    # it returns a string (SERVER FAIL or NOT AVAILABLE)
                    # if it's a SERVER FAIL, we retry soon it's reccurent
                    # if it's a NOT AVAILABLE, we retry later (data not yet published)
                    if (type(res) is str):
                        
                        if (res != "SERVER FAIL" or retry + 1 < SERVER_FAIL_MAX_RETRY):
                            if (SPECTACULAR_VERBOSE):
                                print("\r", end="")
                                print_error(res)
                                
                            else:
                                print(" "*(len(GRID) - geopoint)+"|", end=" ")
                                print_error(res)
                        
                        if (res != "SERVER FAIL"):
                            return res
                        else:
                            retry += 1
                            if (retry >= SERVER_FAIL_MAX_RETRY):
                                if (SPECTACULAR_VERBOSE):
                                    print("\r", end="")
                                    print_error("TOO MANY RETRY")
                                else:
                                    print(" "*(len(GRID) - geopoint)+"|", end=" ")
                                    print_error("TOO MANY RETRY")
                            else:
                                time.sleep(SERVER_FAIL_RETRY)
                                if (SPECTACULAR_VERBOSE):
                                    print("\r", end="")
                                else:
                                    print(f"{c+1:2.0f}/{len(vars_names):2.0f} Getting {' ' * (max_length-len(name))}{name} : |{'='*geopoint}",sep="", end="", flush=True)
                    else:
                        if (SPECTACULAR_VERBOSE and geopoint + 1 < len(GRID)):
                            print("\r", end="")
                            print(f"{c+1:2.0f}/{len(vars_names):2.0f} {name}_{geopoint+2} |", end="")
                            for gp in range(0, geopoint+1):
                                print("=", end="")
                            for gp in range(geopoint+1, len(GRID)):
                                print(" ", end="")
                            print("|", end="")
                            print(f" {geopoint+1}/{len(GRID)}", end="")
                            print(f" Finished in : {duration_to_string(estimated_remaining)}", end=" "*20, flush=True)
                        elif not(SPECTACULAR_VERBOSE):
                            print("=", end="", flush=True)
                    
                        # else, it return the column, and we can gather the next one independently
                        cols.append(res)
                        cols_names.append(name+"_"+str(geopoint+1))
                        retry = SERVER_FAIL_MAX_RETRY+1 # stop while

                if (retry == SERVER_FAIL_MAX_RETRY):
                    
                    return "TOO MANY RETRY"
                
            if (SPECTACULAR_VERBOSE):
                print("\r", end="")
                print(f"{c+1:2.0f}/{len(vars_names):2.0f} {name}:{' ' * (max_length-len(name))} Downloaded in {duration_to_string(time.time() - feature_start)}", end="")
                print(" "*60, flush=True, end="\n")
            else:
                print(f"| Done in : {duration_to_string(time.time() - feature_start)}", flush=True, end="\n")
            
    except Exception as e:
        # print error message on stderr
        e = str(e)
        if (STD_ERR is not None):
            print("\n\n###ERROR: ###\n\n", e, traceback.format_exc(), file=STD_ERR, flush=True)

        # if the error is -70, it means that the URL is invalid
        # (because the data is not available yet)
        if ("[Errno -70] NetCDF: DAP server error:" in e):
            print_error("NOT AVAILABLE")
            return "NOT AVAILABLE"
        
        # else, it's a server fail, we retry
        print_error("SERVER FAIL")
        return "SERVER FAIL"

    # convert each column to a pandas dataframe
    for i in range(len(cols)):
        cols[i] = pd.DataFrame(cols[i])
        cols[i].columns = [cols_names[i]]
    
    # data col
    DATE=pd.DataFrame(np.array(dates), columns=["Date"])
    # date format : YYYYMMDD:HH
    DATE["Date"] = DATE["Date"].apply(lambda x: x.strftime('%Y%m%d:%H'))

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
        last_success = f.read()
        f.close()

    last_success = datetime.datetime.strptime(last_success, '%Y-%m-%d %H:%M:%S')
    last_success = datetime.datetime(last_success.year, last_success.month, last_success.day, last_success.hour)

    # if today - last_success > 16 days, last_success = today - 16 days
    if (today - last_success > datetime.timedelta(days=16)):
        last_success = today - datetime.timedelta(days=16)
        with open('last_success.txt', 'w') as f:
            f.write(str(last_success))
            f.close()
    print(last_success)
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
            print("Gathering data for : ", date.strftime('%Y-%m-%d %H:%M:%S'), "(Remaining : "+str(len(dates) - success_count)+")", flush=True)
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
            print(f"retry in {math.ceil(ALL_GATHERED_RETRY / 60 / 60)}h", flush=True, end=" ")
            time.sleep(ALL_GATHERED_RETRY)
            print("\n\n", flush=True)

        if (type(data) is str):

            if (data == "SERVER FAIL"):
                # if server is temporarily down, retry in 2 min
                time.sleep(SERVER_FAIL_RETRY)
                print("\n\n", flush=True)

            elif (data == "NOT AVAILABLE"):
                # data has not been published yet, retry in 2h
                time.sleep(NOT_AVAILABLE_RETRY)
                print("\n\n", flush=True)

            elif (data == "TOO MANY RETRY"):
                # server is down for longer time, retry in 30 min
                time.sleep(SERVER_FAIL_BREAK)
                print("\n\n", flush=True)


# startup !
GEOS_scrapper()