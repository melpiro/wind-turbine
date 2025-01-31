
from catboost import CatBoostRegressor
from B_Model.AbstractModel import Model as AbstactModel
import numpy as np
import os
from keras.losses import MeanSquaredError


class Model(AbstactModel):
    """
    LSTM model for predicting wind turbine energy production

    Parameters:
    ------------

    CTX: dict
        The hyperparameters context


    Attributes:
    ------------

    name: str (MENDATORY)
        The name of the model for mlflow logs
    
    Methods:
    ---------

    predict(x): (MENDATORY)
        return the prediction of the model

    compute_loss(x, y): (MENDATORY)
        return the loss and the prediction associated to x, y and y_

    training_step(x, y): (MENDATORY)
        do one training step.
        return the loss and the prediction of the model for this batch

    visualize(save_path):
        Generate a visualization of the model's architecture
    """

    name = "CatBoost"

    def __init__(self, CTX:dict):
        """ 
        Generate model architecture
        Define loss function
        Define optimizer
        """
        self.cb = CatBoostRegressor(n_estimators=CTX["N_ESTIMATORS"], learning_rate=CTX["LEARNING_RATE"],random_seed=42,reg_lambda=0.1)
        self.loss = MeanSquaredError()

    def predict(self, x):
        """
        Make prediction for x 
        """
        x = x.reshape(x.shape[0],-1)
        y_ = self.cb.predict(x)
        return y_.reshape(-1,1)

    def compute_loss(self, x, y):
        """
        Make a prediction and compute the loss
        that will be used for training
        """
        y_ = self.predict(x)
        return self.loss(y_, y), y_

    def training_step(self, x, y):
        """
        Do one forward pass and gradient descent
        for the given batch
        """
        x = x.reshape(x.shape[0],-1)
        y= y.reshape(-1)
        self.cb.fit(x, y, verbose=False)
        loss, out = self.compute_loss(x, y)
        return loss, out



    def visualize(self, save_path="./_Artefact/"):
        """
        Generate a visualization of the model's architecture
        """
        pass



    def get_variables(self):
        """
        Return the variables of the model
        """
        return self.cb

    def set_variables(self, variables):
        """
        Set the variables of the model
        """
        self.cb = variables
