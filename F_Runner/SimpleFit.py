



# Mlflow logging
import _Utils.mlflow as mlflow

# Convert CTX to dict for logging hyperparameters
from _Utils.module import module_to_dict 

# For auto-completion, we use Abstract class as virtual type
from B_Model.AbstractModel import Model as _Model_
from E_Trainer.AbstractTrainer import Trainer as _Trainer_



def simple_fit(Model:"type[_Model_]", Trainer:"type[_Trainer_]", CTX, default_CTX=None, experiment_name:str = None):


    # Init mlflow
    run_number = mlflow.init_ml_flow(experiment_name)
    run_name = str(run_number) + " - " + Model.name
    print("Run name : ", run_name)

    # Convert CTX to dict and log it
    CTX = module_to_dict(CTX)
    if (default_CTX != None):
        default_CTX = module_to_dict(default_CTX)
        for param in default_CTX:
            if (param not in CTX):
                CTX[param] = default_CTX[param]
                
    if (CTX["SLIDING_WINDOW"] and "E53 Power [kW]" in CTX["USED_FEATURES"]):
        raise ValueError("SLIDING_WINDOW is not compatible with 'E53 Power [kW]' in USED_FEATURES")


    with mlflow.start_run(run_name=run_name) as run:
        for param in CTX:
            if (type(CTX[param]) == bool): # Convert bool to int the make boolean hyperparameters visualisable in mlflow
                mlflow.log_param(param, int(CTX[param]))
            else:
                mlflow.log_param(param, CTX[param])

        
        # Instanciate the trainer and go !
        trainer = Trainer(CTX, Model)
        metrics = trainer.run()

        # Log the result metrics to mlflow
        for metric_label in metrics:
            value = metrics[metric_label]
            mlflow.log_metric(metric_label, value)

