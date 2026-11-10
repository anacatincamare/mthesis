import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_waiting_time_distribution(toas, period = False, bins = 100, title = "", color = "magenta"):
    """
    Creates a histplot with rugs for the toas. Optional: plots the period as a vertical line.
    """
    gaps = np.diff(toas)
    ax1 = sns.histplot(gaps, bins = bins, stat = "density", alpha = 0.5, color = color)
    if period:
        ax1.vlines(x = period, ymin = 0, ymax = 0.5, color = "black", linestyles="--", alpha = 0.5)
    ax1 = sns.rugplot(gaps, color = "magenta", height = -0.03, clip_on = False)
    ax1.set_title(title)
    return ax1


