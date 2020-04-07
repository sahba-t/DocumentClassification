import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def plot_batch(df, name, x_col="lambda", y_col="accuracy"):
    plt.figure()
    plt.rcParams.update({'font.size': 14})
    for eta in df['eta'].unique():
        eta_df = df[df['eta'] == eta]
        xs = eta_df['lambda']
        ys = eta_df['accuracy'] * 100
        plt.xlabel("Lambda")
        plt.ylabel("Accuracy (%)")
        plt.plot(xs, ys, "o-", label="eta = %.3f" % eta)
    plt.legend()
    plt.savefig(F"../res/{name}", bbox_inches='tight', pad_inches=0.3)
    plt.show()

def plot_progress(df, name):
    plt.figure()
    plt.rcParams.update({'font.size': 14})
    for eta in df['eta'].unique():
        eta_df = df[df['eta'] == eta]
        xs = eta_df['iteration']
        ys = eta_df['accuracy'] * 100
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy (%)")
        plt.plot(xs, ys, "-", label="eta = %.3f" % eta)
    plt.legend()
    plt.savefig(F"../res/{name}", bbox_inches='tight', pad_inches=0.3)
    plt.show()


ETA_BATCH_THRESHOLD1 = 0.004
ETA_BATCH_THRESHOLD2 = 0.007

#param_df = pd.read_csv("../res/par_sweep.csv")
param_df = pd.read_csv("../res/progress.csv")
print(param_df.head())

current_batch = param_df[param_df['eta'] <= ETA_BATCH_THRESHOLD1]
#plot_batch(current_batch, "lower_half.png")
plot_progress(current_batch, "progress_lower.png")

param_df = param_df[param_df['eta'] > ETA_BATCH_THRESHOLD1]
current_batch = param_df[param_df['eta'] <= ETA_BATCH_THRESHOLD2]
#plot_batch(current_batch, "upper_half.png")
plot_progress(current_batch, "progress_mid.png")


param_df = param_df[param_df['eta'] > ETA_BATCH_THRESHOLD2]
#plot_batch(current_batch, "upper_half.png")
plot_progress(param_df, "progress_upper.png")

param_df = pd.read_csv("../res/par_sweep.csv")
plot_batch(param_df[param_df['eta'] <= 0.005], "lower_half.png")
plot_batch(param_df[param_df['eta'] > 0.005], "upper_half.png")