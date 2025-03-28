import os
import numpy as np
import matplotlib.pyplot as plt

def plot_curves(log_dir, task_name):
    files = sorted(f for f in os.listdir(log_dir) if f.endswith("_loss.npy") and (task_name in f))

    for fname in files:
        path = os.path.join(log_dir, fname)
        losses = np.load(path)
        label = fname.replace("_loss.npy", "")
        plt.plot(losses, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curves from All Workers")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/scratch/gpfs/zs8839/cos5682/cos568proj2/fig/{task_name}_loss.pdf")

plot_curves(log_dir = "/scratch/gpfs/zs8839/cos5682/cos568proj2/tmp", task_name="task2a")
#plot_curves(log_dir = "/scratch/gpfs/zs8839/cos5682/cos568proj2/tmp", task_name="task2b")
#plot_curves(log_dir = "/scratch/gpfs/zs8839/cos5682/cos568proj2/tmp", task_name="task3")