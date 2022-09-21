"""
Plotting utilities
"""
from matplotlib import pyplot as plt

def plot_task_error(
        task_num, val_error, 
        boundaries=[], 
        save=True,
        strategy='example',
    ):
    plt.figure(task_num)
    plt.title(f'Error for Task {task_num}')
    plt.plot(val_error[task_num], label=strategy)
    plt.xlabel("iterations")
    plt.ylabel("error")
    for x in boundaries:
        plt.vlines(x, ymin=0, ymax=1, color='r', linestyles="dashed")
    
    if save:
        plt.savefig(f'{strategy}.jpg')