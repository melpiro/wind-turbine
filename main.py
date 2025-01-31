import sys
#############################
# Choose your model here    #
#############################
model = "CatBoost"
#############################

# if args[1] => model = args[1]

if len(sys.argv) > 1:
    model = sys.argv[1]


if model== "CNN":
    import G_Main.EnergyPrediction.exp_CNN as CNN
    CNN.__main__()

elif model== "CatBoost":
    import G_Main.EnergyPrediction.exp_CatBoost as CatBoost
    CatBoost.__main__()

elif model== "LSTM":
    import G_Main.EnergyPrediction.exp_LSTM as LSTM
    LSTM.__main__()

elif model== "Reservoir":
    import G_Main.EnergyPrediction.exp_Reservoir as Reservoir
    Reservoir.__main__()

elif model== "RLinear":
    import G_Main.EnergyPrediction.exp_RLinear as RLinear
    RLinear.__main__()

elif model== "NLinear":
    import G_Main.EnergyPrediction.exp_NLinear as NLinear
    NLinear.__main__()

elif model== "DenseRMoK":
    import G_Main.EnergyPrediction.exp_DenseRMoK as DenseRMoK
    DenseRMoK.__main__()

elif model== "CNN_LSTM":
    import G_Main.EnergyPrediction.exp_CNN_LSTM as CNN_LSTM
    CNN_LSTM.__main__()
    
elif model== "Transformer":
    import G_Main.EnergyPrediction.exp_Transformer as Transformer
    Transformer.__main__()
    
elif model== "Autoformer":
    import G_Main.EnergyPrediction.exp_Autoformer as Autoformer
    Autoformer.__main__()
    
else:
    print("Model not found")
    print("Available models are: CNN, CatBoost, LSTM, Reservoir, RLinear, NLinear, DenseRMoK")
    print("Please choose a model from the list above")
    sys.exit(1)

