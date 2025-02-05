
# Wind turbine prediction

The following code trains machine learning models to predict the power output of a wind turbine. The predictions are based on weather data, such as MERRA-2 data or GEOS forecasts.

## 0. Installation

Run all the pip commands in the file `requirements.txt` to install the required packages. (maybe some of them are missing)


## 1. GEOS Dataset generation

The dataset generation uses GEOS forecasts (UTC timezone) and wind turbine power (CET timezone) to generate UTC datasets for training and evaluation.
For precise evaluation, keep at least 2 months of data.
For example, if you have data from 01/2024 to 12/2024, you can use data from 01/2024 to 10/2024 for training and 11/2024 to 12/2024 for evaluation. The largest the Training dataset, the better the model will be trained. The largest the Evaluation dataset, the more accurate the evaluation will be.

Prepare the data:

- Enter the folder `_gen_geos_dataset/`. 
- Put the raw geos forecasts in the folder `geos_forecasts_utc/`.
- Put wind turbine power output data in the folder `power_cet/`.

### 1.1. Generate training dataset

- Open the file `gen_geos_train.py` 
- Set the parameter `POWER_FILE` to `power_cet/[power_file].csv`
- Set the parameter `BEFORE` the last date of the training dataset __excluded__ (e.g. 20241101:00 to stop at the end of 10/2024)
- Run the script with 
```bash 
python gen_geos_train.py
```
- The script will generate a file names `geos_{start}_{end}_s0.csv`. Copy this file in the folder `A_Dataset/EnergyPrediction/`

### 1.2. Generate evaluation dataset

- Open the file `gen_geos_eval.py`
- Set the parameter `POWER_FILE` to `power_cet/[power_file].csv`
- Set the parameter `AFTER` the first date of the training dataset __included__ (e.g. 20241101:00 to start at the beginning of 11/2024)
- Run the script with 
```bash
python gen_geos_eval.py
```
- The script will generate a folder named `eval_geos_{start}_{end}_s0`. Copy this folder in the folder `A_Dataset/EnergyPrediction/`

## 2. Training models

### 2.1. Configure the used datasets

- Open the file `E_Trainer/EnergyPrediction/Trainer.py` 
- Set the parameter `TRAIN` to either:
    - The file generated in 1.1 `A_Dataset/EnergyPrediction/geos_{start}_{end}_s0.csv` to train the model on geos data
    - The file `./A_Dataset/EnergyPrediction/merra_2014_2020_s0.csv` to train the model on merra data
- Set the parameter `EVAL` to the folder generated in 1.2 `A_Dataset/EnergyPrediction/eval_geos_{start}_{end}_s0` to evaluate the model on geos data 

### 2.2. Choose the model

- Open the file `main.py` at the root of the project
- Change the parameter `model` to the desired model:
    - `CNN'`
    - `CatBoost`
    - `LSTM`
    - `Transformer`
    - ...

### 2.3. Optional: change the hyperparameters

You can modify the parameters of a model in `C_Constants/EnergyPrediction/{model}.py` (e.g. `C_Constants/EnergyPrediction/CNN.py`)

### 2.4. Run the code

- Open a terminal at the root of the project and run the following command:
```bash
python main.py
```

## 3. Results

All the results are saved in the folder `_Artifacts/`.

### 3.1 General results (Run logs)

- Enter the folder `_Artifacts/`
- Open the file `log.txt` 

The file contains two tables. The first one show the best models grouped by training and evaluation datasets. The second one show all the runs.
The RMSE and MAE are optained on the evaluation dataset.
The others columns are the parameters of the model.

### 3.2 Results of a specific model

- Enter the folder `_Artifacts/{model}/` to see the results of a specific model (e.g. `_Artifacts/CNN/`). You can find the following files:
    - `{model}_prediction.pdf`: The plots of the predictions on the evaluation dataset
    - `{model}_prediction/`: A folder containing exel sheets with the predictions and the real values for each evaluation files.

    Other less important files are also saved in the folder as:

    - `loss.png`, `rmse.png`, `mae.png`: The plots of the loss, RMSE and MAE during the training
    - `test_model_while_training.png`: A plot that updates at each epoch during the training to show if the model does well on the training dataset
    - `{model}.png`: The architecture of the model
    - other files are weights of the model, do not delete them if you want to keep the model saved.

