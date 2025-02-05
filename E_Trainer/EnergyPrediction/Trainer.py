
import os
import datetime
import pandas as pd
from numpy_typing import np, ax
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages

import _Utils.mlflow as mlflow
import _Utils.Metrics as Metrics
from   _Utils.save import write, load
import _Utils.Color as C
from   _Utils.Chrono import Chrono
from   _Utils.Color import prntC
from   _Utils.ProgressBar import ProgressBar
from   _Utils.RunLogger import RunLogger



from B_Model.AbstractModel import Model as _Model_
from D_DataLoader.EnergyPrediction.DataLoader import DataLoader
from E_Trainer.AbstractTrainer import Trainer as AbstractTrainer

# |====================================================================================================================
# | CONSTANTS
# |====================================================================================================================

ARTIFACTS = "./_Artifacts/"

# TRAIN = "./A_Dataset/EnergyPrediction/merra_2014_2020_s0.csv"
# TRAIN = "./A_Dataset/EnergyPrediction/merra-04-2023_08-2023_s0.csv"
TRAIN = "./A_Dataset/EnergyPrediction/geos-04-2023_08-2023_s0.csv"

# EVAL = "./A_Dataset/EnergyPrediction/eval_geos_04-2023_09-2023_s0/"
# EVAL = "./A_Dataset/EnergyPrediction/eval_geos_09-2023_09-2023_s0/"
EVAL = "./A_Dataset/EnergyPrediction/eval_geos_24h_09-2023_09-2023/"


H_TRAIN_LOSS = 0
H_TEST_LOSS  = 1
H_TRAIN_RMSE = 2
H_TEST_RMSE  = 3

H_TRAIN_MAE  = 4
H_TEST_MAE   = 5


# for evaluation
METRICS_FUNC = {"rmse":Metrics.RMSE, "mae":Metrics.MAE}
RMSE_LEVEL = [24, 48, 72, 1000]


# |====================================================================================================================
# | GLOBAL VARIABLES
# |====================================================================================================================

BAR = ProgressBar()
RUN_LOGGER = RunLogger("./_Artifacts/logs.pkl")
CHRONO = Chrono()


class Trainer(AbstractTrainer):

    def __init__(self, CTX:dict, Model:"type[_Model_]"):
        super().__init__(CTX, Model)
        self.CTX = CTX

        self.dl = DataLoader(CTX, TRAIN)
        self.model:_Model_ = Model(CTX)
        
        # If "_Artifacts/" folder doesn't exist, create it.
        self.__makes_artifacts__()
        
        try:
            self.model.visualize(save_path=self.ARTIFACTS+"/")
        except Exception as e:
            print("WARNING : visualization of the model failed", e)
            
        # Private attributes
        self.__ep__ = -1
        self.__history__ = None
        self.__history_mov_avg__ = None
        
        

    def __makes_artifacts__(self) -> None:
        self.ARTIFACTS = ARTIFACTS+self.model.name

        if not os.path.exists(ARTIFACTS):
            os.makedirs(ARTIFACTS)
        if not os.path.exists(self.ARTIFACTS):
            os.makedirs(self.ARTIFACTS)
        if not os.path.exists(self.ARTIFACTS+"/weights"):
            os.makedirs(self.ARTIFACTS+"/weights")
            
        # remove test_model_while_training_ep_*.png
        os.system("rm "+self.ARTIFACTS+"/test_model_while_training_ep_*.png")
        
# |====================================================================================================================
# | TRAINING
# |====================================================================================================================

    def train(self):
        
        prntC(C.BLUE, "Training :")

        for ep in range(1, self.CTX["EPOCHS"] + 1):
            
            CHRONO.start()
            
            # batchs
            x_train, y_train = self.dl.get_train(self.CTX["NB_BATCH"], self.CTX["BATCH_SIZE"])
            y_train_ = np.empty(y_train.shape)
            x_test, y_test = self.dl.get_test()
            y_test_ = np.empty(y_test.shape)
            
            BAR.reset(max=len(x_train) + len(x_test))
            
            # training
            for batch in range(len(x_train)):
                _,  y_train_[batch] = self.model.training_step(x_train[batch], y_train[batch])
                BAR.update()
                
            # testing
            for batch in range(len(x_test)):
                loss, y_test_[batch] = self.model.compute_loss(x_test[batch], y_test[batch])
                BAR.update()
                
            self.__epoch_stats__(ep, y_train, y_train_, y_test, y_test_)
            
        self.__load_best_model__()
        
# |--------------------------------------------------------------------------------------------------------------------
# |    STATISTICS FOR TRAINING
# |--------------------------------------------------------------------------------------------------------------------
    
    def __epoch_stats__(self, ep:int,
                        y_train:np.float64_3d[ax.batch, ax.sample, ax.feature],
                        y_train_:np.float64_3d[ax.batch, ax.sample, ax.feature],\
                        y_test:np.float64_3d[ax.batch, ax.sample, ax.feature], \
                        y_test_:np.float64_3d[ax.batch, ax.sample, ax.feature]):
        
        train_loss, train_rmse, train_mae = self.__prediction_statistics__(y_train, y_train_)
        test_loss,  test_rmse,  test_mae  = self.__prediction_statistics__(y_test, y_test_)
        

        self.__loss_curves__(ep, train_loss, test_loss, train_rmse, test_rmse, train_mae, test_mae)
        self.__plot_epoch_stats__()

        self.__print_epoch_stats__(ep, train_loss, test_loss, train_rmse, test_rmse, train_mae, test_mae)
        
        self.__plot_train_exemple__(ep, y_test[0, :512, :], y_test_[0, :512, :])
    
        # save the model
        write(self.ARTIFACTS+"/weights/"+self.model.name+"_"+str(ep)+".w", self.model.get_variables())
        
    
    
    def __prediction_statistics__(self,
                                y:np.float64_3d[ax.batch, ax.sample, ax.feature],
                                y_:np.float64_3d[ax.batch, ax.sample, ax.feature],)\
                        -> "tuple[float, float]":

        loss, rmse, mae = 0, 0, 0
        for batch in range(len(y)):      
            
            loss += Metrics.MSE(y[batch], y_[batch])
              
            y_[batch] = self.dl.yScaler.inverse_transform(y_[batch])
            y_[batch][y_[batch] < 0] = 0
            y[batch] = self.dl.yScaler.inverse_transform(y[batch])
            
            rmse += Metrics.RMSE(y[batch], y_[batch])
            mae += Metrics.MAE(y[batch], y_[batch])
            
        return loss/len(y), rmse/len(y), mae/len(y)
    
    
    
    def __loss_curves__(self, ep:int,
                        train_loss:float, test_loss:float,
                        train_rmse:float, test_rmse,
                        train_mae:float, test_mae) -> None:
        # On first epoch, initialize loss curves
        if (self.__ep__ == -1 or self.__ep__ > ep):
            self.__history__         = np.full((6, self.CTX["EPOCHS"]), np.nan, dtype=np.float64)
            self.__history_mov_avg__ = np.full((6, self.CTX["EPOCHS"]), np.nan, dtype=np.float64)
            
        # Save epoch statistics
        self.__ep__ = ep
        self.__history__[:, ep-1] = [train_loss, test_loss,
                                     train_rmse, test_rmse,
                                     train_mae,  test_mae]
        
        for i in range(len(self.__history__)):
            self.__history_mov_avg__[i, ep-1] = Metrics.moving_average_at(self.__history__[i], ep-1, w=5)
            
    
    
    def __print_epoch_stats__(self, ep:int,
                            train_loss:float, test_loss:float,
                            train_rmse:float, test_rmse:float,
                            train_mae:float, test_mae:float) -> None:
    

        prntC(C.INFO,  "Epoch :",      C.BLUE, ep, C.RESET, "/", C.BLUE, self.CTX["EPOCHS"], C.RESET,
                     "- Takes :",      C.BLUE, CHRONO)
        prntC(C.INFO_, "Train mse :" , C.BLUE, round(train_loss, 4), C.RESET,
                     "- Test  mse :" , C.BLUE, round(test_loss,  4))
        prntC(C.INFO_, "Train MAE  :", C.BLUE, round(train_mae, 1), C.RESET,
                     "- Test  MAE  :", C.BLUE, round(test_mae,  1))
        prntC(C.INFO_, "Train RMSE :", C.BLUE, round(train_rmse, 1), C.RESET,
                     "- Test  RMSE :", C.BLUE, round(test_rmse,  1))
        prntC()
        
        
        
    def __plot_epoch_stats__(self) -> None:

        # plot loss curves
        Metrics.plot_loss(self.__history__[H_TRAIN_LOSS], self.__history__[H_TEST_LOSS],
                         self.__history_mov_avg__[H_TRAIN_LOSS], self.__history_mov_avg__[H_TEST_LOSS],
                            type="loss", path=self.ARTIFACTS+"/loss.png")

        Metrics.plot_loss(self.__history__[H_TRAIN_RMSE], self.__history__[H_TEST_RMSE],
                         self.__history_mov_avg__[H_TRAIN_RMSE], self.__history_mov_avg__[H_TEST_RMSE],
                            type="rmse", path=self.ARTIFACTS+"/rmse.png")
        
        Metrics.plot_loss(self.__history__[H_TRAIN_MAE], self.__history__[H_TEST_MAE],
                        self.__history_mov_avg__[H_TRAIN_MAE], self.__history_mov_avg__[H_TEST_MAE],
                                type="mae", path=self.ARTIFACTS+"/mae.png")

        
        
    def __plot_train_exemple__(self, ep, y, y_):
        
        FIGS = 3
        F0 = 0
        
        if (self.CTX["SLIDING_WINDOW"]):
            y_true = y[:, 0]
            if (self.CTX["LOOK_AHEAD"]>1):
                y_true = np.concatenate([y[:, 0], y[-self.CTX["LOOK_AHEAD"]+1:, -1]])
                
            y_preds = np.full((len(y_true), self.CTX["LOOK_AHEAD"]), np.nan)
            for i in range(self.CTX["LOOK_AHEAD"]):
                y_preds[i:i+len(y_), i] = y_[:, i]
        else:
            FIGS = 1
            y_true = y[::self.CTX["LOOK_AHEAD"], :].reshape(-1)
            y_preds = y_[::self.CTX["LOOK_AHEAD"], :].reshape(-1)
            
            
        fig, ax = plt.subplots(FIGS, 1, figsize=(10, 4*FIGS))
        if (FIGS == 1): ax = [ax]
        
        if (self.CTX["SLIDING_WINDOW"]):
            # plot all
            ax[F0+0].plot(y_true, label="true")
            ax[F0+0].plot(y_preds, label="predicted")
            
            # plot mean
            y_mean = np.nanmean(y_preds, axis=1)
            ax[F0+1].plot(y_true, label="true")
            ax[F0+1].plot(y_mean, label="predicted")
            
            # plot zone
            y_sorted = np.sort(y_preds, axis=1)
            ax[F0+2].plot(y_true, label="true")
            nb_zones, shift = divmod(self.CTX["LOOK_AHEAD"]+1,2)
            for i in range(nb_zones):
                mid = nb_zones-1
                ax[F0+2].fill_between(range(len(y_true)), y_sorted[:, mid-i], y_sorted[:, mid+i+shift], alpha=0.1, label="predicted", color="tab:blue")
        
        else:
            ax[F0].plot(y_true, label="true")
            ax[F0].plot(y_preds, label="predicted")
            # show legend
            ax[F0].legend()    
        
        #tight layout
        plt.tight_layout()
        
        fig.savefig(self.ARTIFACTS+"/test_model_while_training.png")
        plt.close()
        
        
        
    def __load_best_model__(self):
        # # load back best model
        if (len(self.__history__[H_TEST_RMSE]) > 0):
            # find model with the "best" test rmse
            best_i = np.argmin(self.__history_mov_avg__[H_TEST_RMSE]) + 1

            print("load best model, epoch : ", best_i, " with rmse : ", self.__history__[H_TEST_RMSE][best_i-1], flush=True)
            
            best_variables = load(self.ARTIFACTS+"/weights/"+self.model.name+"_"+str(best_i)+".w")
            self.model.set_variables(best_variables)
        else:
            print("WARNING : no history of training has been saved")

        # save parameters
        write(self.ARTIFACTS+"/"+self.model.name+".w", self.model.get_variables())
        write(self.ARTIFACTS+"/"+self.model.name+".xs", self.dl.xScaler.get_variables())
        write(self.ARTIFACTS+"/"+self.model.name+".ys", self.dl.yScaler.get_variables())
        
# |====================================================================================================================
# | EVALUATION
# |====================================================================================================================

    def load(self):
        self.model.set_variables(load(self.ARTIFACTS+"/"+self.model.name+".w"))
        self.dl.xScaler.set_variables(load(self.ARTIFACTS+"/"+self.model.name+".xs"))
        self.dl.yScaler.set_variables(load(self.ARTIFACTS+"/"+self.model.name+".ys"))
        

            
    def eval(self):
        prntC(C.BLUE, "Evaluation :")
        
        true_power = pd.read_excel(EVAL+"TruePower.xlsx")
        files = self.__list_eval_files__()
        
        BAR.reset(0, len(files))
        
        pdf = PdfPages(self.ARTIFACTS+"/"+self.model.name+"_prediction.pdf")
        metrics = np.full((len(files), len(RMSE_LEVEL), len(METRICS_FUNC)), np.nan)
        for f in range(len(files)):
            file = files[f]
            
            # predict
            x, _, df, start_i = self.dl.genEval(file)
            y_ = self.model.predict(x[0])
            y_ = self.dl.yScaler.inverse_transform(y_)
            y_[y_ < 0] = 0
            
            metrics[f] = self.__eval_stats__(file, y_, df, true_power, pdf)
            
            BAR.update()
        pdf.close()

        return self.__log_metrics__(metrics)
    
        
        
    def __list_eval_files__(self) -> "list[str]":
        files = os.listdir(EVAL)
        return np.sort([EVAL+f for f in files if f.endswith(".xlsx") and f.startswith("in")])
    
    
                
    def __eval_stats__(self, file:str,
                             y_:np.float32_3d[ax.batch, ax.time, ax.feature],
                             df:pd.DataFrame, true_power:pd.DataFrame, pdf:PdfPages,
                             ) -> "np.float32_2d[ax.time, ax.feature]":
        

        start_i = 24
        
        if (self.CTX["SLIDING_WINDOW"]):
            # re align y_ with true power
            y_sample_:np.float32_2d[ax.time, ax.feature] = np.full((len(df) - start_i, self.CTX["LOOK_AHEAD"]), np.nan)
            for i in range(self.CTX["LOOK_AHEAD"]):
                y_sample_[i:i+len(y_), i] = y_[:, i]
            y_mean_ = np.nanmean(y_sample_, axis=1).reshape(-1, 1)
        else:
            y_sample_ = np.full((len(df) - start_i, 1), np.nan)
            y_sample_[:self.CTX["LOOK_AHEAD"], 0] = y_[0]
            y_mean_ = y_sample_
            
        # get y
        starting_date = df["Date:time [YYYMMDD:HH]"][start_i]
        start_index = true_power[true_power["Date"] == starting_date].index[0]
        y = true_power["True power"][start_index:start_index+len(y_sample_)].values.reshape(-1, 1)
        
        # compute metrics
        metrics = np.full((len(RMSE_LEVEL), len(METRICS_FUNC)), np.nan)
        metrics_per_lookahead = np.full((self.CTX["LOOK_AHEAD"], len(RMSE_LEVEL), len(METRICS_FUNC)), np.nan)
        for l, level in enumerate(RMSE_LEVEL):
            for i, metric_func in enumerate(METRICS_FUNC.values()):
                if (self.CTX["SLIDING_WINDOW"] or level <= self.CTX["LOOK_AHEAD"]):
                    metrics[l, i] = metric_func(y[:level], y_mean_[:level])
                    metrics_per_lookahead[:, l, i] = [metric_func(y[:level], y_sample_[:level, s:s+1]) for s in range(y_sample_.shape[1])]

        self.__save_prediction__(df, file, y_mean_, y, start_i)
        self.__plot_eval__(pdf, df, y_mean_, y)
        
        return metrics
    
    
    
    __save_pred_first__ = True
    def __save_prediction__(self, df, f, y_:np.float32_2d[ax.time, ax.feature], y:np.float32_2d[ax.time, ax.feature], start_i:int):        
        if (self.__save_pred_first__):
            self.__save_pred_first__ = False
            # create artifacts folder
            if not(os.path.exists(self.ARTIFACTS+"/"+self.model.name+"_prediction/")):
                os.makedirs(self.ARTIFACTS+"/"+self.model.name+"_prediction/")
            else:
                os.system("rm "+self.ARTIFACTS+"/"+self.model.name+"_prediction/*")
                
        out_df = pd.DataFrame()
        out_df["Date:time [YYYMMDD:HH]"] = df["Date:time [YYYMMDD:HH]"][start_i-self.CTX["HISTORY"]:start_i+len(y_)]
        out_df["Predicted power"] = np.concatenate([np.zeros(self.CTX["HISTORY"]), y_[:, 0]])
        out_df["True power"] = np.concatenate([np.zeros(self.CTX["HISTORY"]), y[:, 0]])
        out_df.to_excel(self.ARTIFACTS+"/"+self.model.name+"_prediction/"+f.split("/")[-1], index=False)
    
    
    def __plot_eval__(self, pdf, df, y_:np.float32_2d[ax.time, ax.feature], y:np.float32_2d[ax.time, ax.feature], best_look_ahead:int=None):

        if (not(self.CTX["SLIDING_WINDOW"])):
            y_ = y_[:self.CTX["LOOK_AHEAD"]]
            y  =  y[:self.CTX["LOOK_AHEAD"]]
            
        dates = df["Date:time [YYYMMDD:HH]"][24:len(y_)+24].values
        dates = [date.split(":")[0] for date in dates]
        dates = [datetime.datetime.strptime(date, "%Y%m%d").strftime("%d/%m/%Y") for date in dates]
        
        y_mean = np.nanmean(y_, axis=1)
        y_sorted = np.sort(y_, axis=1)
        nb_zones = (self.CTX["LOOK_AHEAD"]+1)//2
        nb_zones, shift = divmod(self.CTX["LOOK_AHEAD"]+1,2)

        
        plt.figure(figsize=(20, 8))
        plt.title(f"RMSE = {Metrics.RMSE(y, y_mean.reshape(-1, 1)):.2f}", fontsize=20)
        
        if (self.CTX["SLIDING_WINDOW"]):
            for i in range(nb_zones):
                mid = nb_zones-1
                plt.fill_between(range(len(y_sorted)), y_sorted[:, mid-i], y_sorted[:, mid+i+shift], alpha=0.1, label="predicted", color="tab:blue")
        
        plt.plot(y_mean, label="predicted", color="tab:blue", linewidth=2)
        plt.plot(y, label="true", color="tab:green", linewidth=2)
        
        if (best_look_ahead is not None):
            plt.plot(y_[:, best_look_ahead], label="best look ahead", color="tab:red", linewidth=2)
        
        plt.xticks(range(0, len(dates), 24), dates[::24], rotation=0)
        plt.grid()
        plt.legend()
        pdf.savefig()
        plt.close()
        
    
    
    def __log_metrics__(self, metrics:np.float32_3d[ax.sample, ax.time, ax.feature]) -> dict:
        
        # metrics logging
        logs = {"model":self.model.name}
        print("Metrics :")
        for i, m in enumerate(METRICS_FUNC.keys()):
            print(f"\t{m.upper()} :")
            for l, level in enumerate(RMSE_LEVEL):
                if (level == 1000):
                    print(f"\t - {m.upper()}_FULL fcst : {np.mean(metrics[:, l, i]):.2f}")
                    logs[f"{m.upper()}_FULL"] = round(np.mean(metrics[:, l, i]), 1)
                else:
                    print(f"\t - {m.upper()} {level}H fcst : {np.mean(metrics[:, l, i]):.2f}")
                    logs[f"{m.upper()}_{level}"] = round(np.mean(metrics[:, l, i]), 1)
                    
        logs["TRAIN"] = TRAIN.split("/")[-1].split(".")[0]
        logs["EVAL"] = EVAL.split("/")[-2]
        

        if (self.CTX["EPOCHS"] > 0):
            H = "HYPERPARAMETERS"
            groups = {
                "LEARNING_RATE":H,
                "EPOCHS":H,
                "BATCH_SIZE":H,
                "NB_BATCH":H,
                "HISTORY":H,
                "LOOK_AHEAD":H,
                "FEATURES_IN":H,
                "DROPOUT":H,
                "TRAIN":"DATASET",
                "EVAL":"DATASET",
            }
            dtypes = {
                "DATASET":str,
            }
            for i in METRICS_FUNC:
                for l in RMSE_LEVEL:
                    if (l == 1000):
                        groups[f"{i.upper()}_FULL"] = i.upper()
                    else:
                        groups[f"{i.upper()}_{l}"] = i.upper()
                        
                        
            # for each variable in CTX
            exept = ["MAX_BATCH_SIZE", "TEST_RATIO", "INPUT_LEN", "OUTPUT_LEN"]
            for k in self.CTX:
                if (k in exept):
                    continue
                # if the variable is numeric
                if (isinstance(self.CTX[k], (int, float))):
                    # if the variable is not in the groups
                    if (k not in groups):
                        groups[k] = "SPECIFIC HYPERPARAMETERS"
                    else:
                        groups[k] = H
                        
                    logs[k] = self.CTX[k]
                        
            RUN_LOGGER.add_run(logs, groups, dtypes)
            
            file = open(ARTIFACTS+"log.txt", "w")
            
            loggers_ = RUN_LOGGER.split_by("EVAL")
            loggers = []
            for i in range(len(loggers_)):
                loggers.extend(loggers_[i].split_by("TRAIN"))
            for i in range(len(loggers)):
                loggers[i].get_best_groupes_by("RMSE_24", "model", maximize=False)
            logger:RunLogger = RunLogger.join(loggers).group_by("TRAIN").group_by("EVAL")
            logger.render(file, "Best models by dataset")
            file.write("\n\n")
            RUN_LOGGER.render(file, "All runs")
            file.close()
            
            
            
        return logs
        
        
