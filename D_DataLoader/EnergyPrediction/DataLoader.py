 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy_typing import np, ax

from _Utils.Scaler3D import MinMaxScaler2D, StandardScaler3D, StandardScaler2D
import _Utils.Color as C
from _Utils.Color import prntC
from   _Utils.ProgressBar import ProgressBar


import D_DataLoader.Utils as U
import D_DataLoader.EnergyPrediction.Utils as SU
from D_DataLoader.AbstractDataLoader import DataLoader as AbstractDataLoader


# |====================================================================================================================
# | GLOBAL VARIABLES
# |====================================================================================================================

BAR = ProgressBar()

# |====================================================================================================================
# | DATA LOADER
# |====================================================================================================================

# managing the data preprocessing
class DataLoader(AbstractDataLoader):
    
    CTX:dict
    xScaler:StandardScaler3D
    yScaler:StandardScaler2D
    
    x:"np.float32_2d[ax.time, ax.feature]"
    y:"np.float32_2d[ax.time, ax.feature]"
    x_train:"np.float32_2d[ax.time, ax.feature]"
    x_test :"np.float32_2d[ax.time, ax.feature]"
    y_train:"np.float32_2d[ax.time, ax.feature]"
    y_test :"np.float32_2d[ax.time, ax.feature]"

# |====================================================================================================================
# |     INITIALISATION : LOADING RAW DATASET FROM DISK
# |====================================================================================================================

    def __init__(self, CTX, path) -> None:    
        self.CTX = CTX

        self.xScaler = StandardScaler3D()
        self.yScaler = StandardScaler2D()
        
        training = (CTX["EPOCHS"] and path != "")
        if (training):
            self.x, self.y, _ = self.__get_dataset__(path)
            self.x_train, self.y_train, self.x_test, self.y_test = self.__split__(self.x, self.y) 
            
            prntC("Train dataset size :", C.BLUE, len(self.x_train))
            prntC("Test dataset size :", C.BLUE, len(self.x_test))
            
        else:
            self.x, self.y = None, None



    @staticmethod
    def __load_dataset__(CTX, path) -> """tuple[np.float32_2d[ax.time, ax.feature], np.float32_2d[ax.time, ax.feature], pd.DataFrame]""":
        df = U.read_windpower(path)
        return U.df_to_feature_array(CTX, df.copy()) + (df,)
    
# |====================================================================================================================
# |    SCALERS
# |====================================================================================================================

    def __scalers_transform__(self, x_batch:np.float64_3d[ax.sample, ax.time, ax.feature],
                                    y_batch:np.float64_2d[ax.sample, ax.feature]=None) \
            -> """tuple[np.float64_3d[ax.sample, ax.time, ax.feature], np.float64_2d[ax.sample, ax.feature]]
                | np.float64_3d[ax.sample, ax.time, ax.feature]""":

        if (not(self.xScaler.is_fitted())):
            self.xScaler.fit(x_batch)
        x_batch = self.xScaler.transform(x_batch)

        if (y_batch is not None):
            if (not(self.yScaler.is_fitted())):
                self.yScaler.fit(y_batch)

            y_batch = self.yScaler.transform(y_batch)
            return x_batch, y_batch
        return x_batch

# |====================================================================================================================
# |     UTILS
# |====================================================================================================================

    def __reshape__(self, x:np.float32_3d[ax.sample, ax.time, ax.feature], 
                    y:np.float32_2d[ax.sample, ax.feature], 
                    nb_batch:int, batch_size:int) -> """tuple[
            np.float32_4d[ax.batch, ax.sample, ax.time, ax.feature],
            np.float32_3d[ax.batch, ax.sample, ax.feature]]""":

        x_batches = x.reshape(nb_batch, batch_size, self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        y_batches = y.reshape(nb_batch, batch_size, self.CTX["OUTPUT_LEN"])
        
        return x_batches, y_batches
# |====================================================================================================================
# |    GENERATE A TRAINING SET
# |====================================================================================================================

    def get_train(self, nb_batch, batch_size) -> """tuple[
            np.float32_4d[ax.batch, ax.sample, ax.time, ax.feature],
            np.float32_3d[ax.batch, ax.sample, ax.feature]]""":

        CTX = self.CTX

        # Allocate memory for the batches
        x_batch, y_batch = SU.alloc_batch(CTX, nb_batch * batch_size)

        for n in range(len(x_batch)):
            x_sample, y_sample = SU.gen_random_sample(CTX, self.x_train, self.y_train)

            x_batch[n] = x_sample
            y_batch[n] = y_sample

        x_batch, y_batch = self.__scalers_transform__(x_batch, y_batch)
        x_batches, y_batches = self.__reshape__(x_batch, y_batch, nb_batch, batch_size)

        return x_batches, y_batches



    def get_test(self) -> """tuple[
            np.float32_4d[ax.batch, ax.sample, ax.time, ax.feature],
            np.float32_3d[ax.batch, ax.sample, ax.feature]]""":
                
                
        nb_samples = len(self.x_test)-self.CTX["HISTORY"]-self.CTX["LOOK_AHEAD"]+1
        nb_batch = nb_samples // self.CTX["MAX_BATCH_SIZE"]
        if (nb_batch == 0):
            nb_batch = 1
            batch_size = nb_samples
        else:
            nb_samples = nb_batch * self.CTX["MAX_BATCH_SIZE"]
            batch_size = self.CTX["MAX_BATCH_SIZE"]

        
        x_batch, y_batch = SU.alloc_batch(self.CTX, nb_samples)

        for i, t in enumerate(range(self.CTX["HISTORY"], nb_samples+self.CTX["HISTORY"])):
            x_batch[i] = SU.gen_sample(self.CTX, self.x_test, t)
            y_batch[i] = self.y_test[t:self.CTX["LOOK_AHEAD"]+t].reshape(-1)

        x_batch, y_batch = self.__scalers_transform__(x_batch, y_batch)        
        x_batches, y_batches = self.__reshape__(x_batch, y_batch, nb_batch, batch_size)
        return x_batches, y_batches
    
    

    def genEval(self, path) -> """tuple[
            np.float32_4d[ax.batch, ax.sample, ax.time, ax.feature],
            np.float32_3d[ax.batch, ax.sample, ax.feature],
            pd.DataFrame, int]""":
                
        x, y, df = self.__load_dataset__(self.CTX, path)

        start_i = 24
        
        nb_samples = len(x)-self.CTX["LOOK_AHEAD"]+1 - start_i
        if not(self.CTX["SLIDING_WINDOW"]):
            nb_samples = 1
        
        
        x_batch, y_batch = SU.alloc_batch(self.CTX, nb_samples)

        for i, t in enumerate(range(start_i, nb_samples+start_i)):
            x_batch[i] = SU.gen_sample(self.CTX, x, t)
            y_batch[i] = y[t:self.CTX["LOOK_AHEAD"]+t].reshape(-1)
            
        x_batch, y_batch = self.__scalers_transform__(x_batch, y_batch)
        x_batches, y_batches = self.__reshape__(x_batch, y_batch, 1, len(x_batch))
        
        return x_batches, y_batches, df, start_i

