from numpy_typing import np, ax
import pandas as pd
import os
from _Utils import Color

import D_DataLoader.Utils as U




# |====================================================================================================================
# | SAMPLE GENERATION
# |====================================================================================================================


def alloc_sample(CTX:dict)\
        -> "np.float32_2d[ax.time, ax.feature]":

    x_sample = np.zeros((CTX["INPUT_LEN"],CTX["FEATURES_IN"]), dtype=np.float32)
    return x_sample

def alloc_batch(CTX:dict, size:int) -> """tuple[
        np.float32_3d[ax.sample, ax.time, ax.feature],
        np.float32_2d[ax.sample, ax.feature]]""":

    x_batch = np.zeros((size, CTX["INPUT_LEN"], CTX["FEATURES_IN"]), dtype=np.float32)
    y_batch = np.zeros((size, CTX["OUTPUT_LEN"]), dtype=np.float32)
    return x_batch, y_batch



def gen_random_sample(CTX:dict, x:np.float32_2d[ax.time, ax.feature], y:np.float32_2d[ax.time, ax.feature])-> """tuple[
        np.float32_2d[ax.time, ax.feature],
        np.float32_2d[ax.feature]]""":
            
    # spike = np.random.randint(0, 100) < 40

    # pick a random location
    t = -1
    while (t == -1 or y[t] == -1):
        t = np.random.randint(CTX["HISTORY"], len(x)-CTX["LOOK_AHEAD"]+1)
        
        # if (spike and y[t] < 400):
        #     t = -1
        

    x = gen_sample(CTX, x, t)
    y = y[t: t + CTX["LOOK_AHEAD"]].reshape(-1)
    
    return x, y

def gen_sample(CTX:dict, x:np.float32_2d[ax.time, ax.feature], t:int) -> """tuple[
        np.float32_2d[ax.time, ax.feature],
        np.float32_2d[ax.feature]]""":
             # extract the sample
             
    start = t - CTX["HISTORY"]
    end = t + CTX["LOOK_AHEAD"]
    # shift = U.compute_shift(start, end, CTX["DILATATION_RATE"])
    x_sample = x[start:end]#:CTX["DILATATION_RATE"]]
    
    power_index = CTX["FEATURE_MAP"].get("E53 Power [kW]", -1)
    if (power_index != -1):
        x_sample[CTX["HISTORY"]:, power_index] = 0
    
    return x_sample