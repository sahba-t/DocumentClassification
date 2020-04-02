import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def plot_batch(df, name):
    plt.figure()
    plt.rcParams.update({'font.size': 14})
    for eta in df['eta'].unique():
        eta_df = df[df['eta'] == eta]
        xs = eta_df['lambda']
        ys = eta_df['accuracy']
        plt.xlabel("Lambda")
        plt.ylabel("Accuracy")
        plt.plot(xs, ys, "o-", label="eta = %.3f" % eta)
    plt.legend()
    plt.savefig(F"../res/{name}", bbox_inches='tight', pad_inches=0.3)
    plt.show()


ETA_BATCH_THRESHOLD = 0.005
param_df = pd.read_csv("../res/par_sweep.csv")
print(param_df.head())

current_batch = param_df[param_df['eta'] <= ETA_BATCH_THRESHOLD]
plot_batch(current_batch, "lower_half.png")

current_batch = param_df[param_df['eta'] > ETA_BATCH_THRESHOLD]
plot_batch(current_batch, "upper_half.png")
