
from B_Model.AbstractModel import Model as AbstactModel

import tensorflow as tf
from keras.layers import *
from reservoirpy.nodes import Reservoir

import numpy as np
import os

import _Utils.Color as C
from   _Utils.Color import prntC


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)



class Model(AbstactModel):

    name = "Reservoir"

    def __init__(self, CTX:dict):

        # load context
        self.CTX = CTX

        # save the number of training steps
        self.nb_train = 0


        # self.reservoir = Reservoir(1000,
        #     lr=0.5, sr=0.9
        # )
        
        self.reservoir = Reservoir(500,
            lr=CTX["lr"], sr=CTX["sr"], 
            input_connectivity = CTX["input_connectivity"], 
            rc_connectivity=CTX["rc_connectivity"]
        )


        x_input_shape = (self.CTX["INPUT_LEN"], self.reservoir.output_dim)
        x = tf.keras.Input(shape=x_input_shape, name='input')
        z= x
        z = Flatten()(z)
        z = Dense(self.CTX["OUTPUT_LEN"], activation="linear")(z)

        self.readout = tf.keras.Model(x, z)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate=CTX["LEARNING_RATE"])



    def __predict_reservoir__(self, x):
        y_ = np.zeros((len(x), self.CTX["INPUT_LEN"], self.reservoir.output_dim))

        for s in range(len(x)):
            # reset the reservoir state
            self.reservoir.reset()
            # give the time series to the reservoir
            for t in range(len(x[s])):
                y_[s, t] = self.reservoir(x[s][t].reshape(1, -1))

        return y_



    def predict(self, x):
        y_reservoir = self.__predict_reservoir__(x)
        return self.readout(y_reservoir).numpy()


    def compute_loss(self, x, y):
        y_pred = self.predict(x)
        return self.loss(y, y_pred), y_pred


    def training_step(self, x, y):
        y_reservoir = self.__predict_reservoir__(x)

        with tf.GradientTape(watch_accessed_variables=True) as tape:
            output = self.readout(y_reservoir)
            loss = self.loss(y, output)

            gradients = tape.gradient(loss, self.readout.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.readout.trainable_variables))

        self.nb_train += 1
        return loss, output.numpy()





    def visualize(self, save_path="./_Artifacts/"):
        """
        Generate a visualization of the model's architecture
        """
        filename = os.path.join(save_path, self.name+".png")
        tf.keras.utils.plot_model(self.readout, to_file=filename, show_shapes=True)



    def get_variables(self):
        """
        Return the variables of the model
        """
        readout_arch = self.readout.trainable_variables
        return self.reservoir, readout_arch

    def set_variables(self, variables):
        """
        Set the variables of the model
        """
        self.reservoir, readout_arch = variables

        for i in range(len(readout_arch)):
            self.readout.trainable_variables[i].assign(readout_arch[i])



