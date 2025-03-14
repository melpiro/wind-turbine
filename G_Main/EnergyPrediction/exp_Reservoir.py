

# Import the model
from B_Model.EnergyPrediction.Reservoir import Model

# Import the context (hyperparameters, constants, etc...)
import C_Constants.EnergyPrediction.Reservoir as CTX
import C_Constants.EnergyPrediction.DefaultCTX as DefaultCTX

# Import the training loop adapted to the model
from E_Trainer.EnergyPrediction.Trainer import Trainer

# Choose the training method
#   * simple_fit: Classical way to fit the model : once
#   * multi_fit: Fit the model multiple times to check the stability (Not implemented yet)
from F_Runner.SimpleFit import *
from F_Runner.ConfirmFit import *

import os



def __main__():
    parent_dir = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
    simple_fit(Model, Trainer, CTX, default_CTX=DefaultCTX, experiment_name=parent_dir)
