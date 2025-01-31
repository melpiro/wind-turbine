
# import numpy as np
from numpy_typing import np, ax

def RMSE(y, y_) -> float:
    y = np.where(y <= 0, y_, y)
    return np.sqrt(np.nanmean((y - y_)**2))

def MAE(y, y_) -> float:
    # remplace y's nan by y_ values
    y = np.where(y <= 0, y_, y)
    return np.nanmean(np.abs(y - y_))

def MSE(y, y_) -> float:
    # y = np.where(y <= 0, y_, y)
    return np.mean((y - y_)**2)




def plot_loss(train:np.ndarray, test:np.ndarray,
             train_avg:np.ndarray, test_avg:np.ndarray,
             type:str="loss", path:str="") -> None:

    # Plot the loss curves
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.grid()

    ax.plot(np.array(train), c="tab:blue", linewidth=0.5)
    ax.plot(np.array(test), c="tab:orange", linewidth=0.5)

    label = "loss"
    if (type != None):
        label = type

    ax.plot(np.array(train_avg), c="tab:blue", ls="--", label=f"train {label}")
    ax.plot(np.array(test_avg), c="tab:orange", ls="--", label=f"test {label}")
    ax.set_ylabel(label)


    ax.set_xlabel("Epoch number")
    # x start at 1 to len(train)
    ax.set_xlim(1, len(train))
    ax.legend()
    fig.savefig(path)
    plt.close(fig)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inv_sigmoid(x):
    return np.log(x / (1 - x))


def moving_average_at(x:np.float64_1d, i:int, w:int) -> float:
    return np.mean(x[max(0, i-w):i+1])

def moving_average(x:np.float64_1d, w:int) -> np.float64_1d:
    r = np.zeros(len(x))
    for i in range(len(x)):
        r[i] = moving_average_at(x, i, w)
    return r