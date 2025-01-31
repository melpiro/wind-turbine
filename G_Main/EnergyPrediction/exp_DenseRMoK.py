

# Import the model
from B_Model.EnergyPrediction.DenseRMoK import Model

# Import the context (hyperparameters, constants, etc...)
import C_Constants.EnergyPrediction.DenseRMoK as CTX
import C_Constants.EnergyPrediction.DefaultCTX as DefaultCTX

# Import the training loop adapted to the model
from E_Trainer.EnergyPrediction.Trainer import Trainer

# Choose the training method
#   * simple_fit: Classical way to fit the model : once
#   * multi_fit: Fit the model multiple times to check the stability (Not implemented yet)
from F_Runner.SimpleFit import *

import os



def __main__():
    simple_fit(Model, Trainer, CTX, default_CTX=DefaultCTX)

