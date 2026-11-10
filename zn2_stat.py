import json as js
import matplotlib.pyplot as plt
import numpy as np
import scipy
from astropy import timeseries
import random
from toa_formatting import toa_to_phase, toa_flatten
from zn_squared import zn_squared


def zn2_stat(source_name, n, lowest_freq = False, highest_freq = False, nr_freq = 5000, plotting = False, saving = False, toa = [], flatten = False):
    """
    The function does the zn^2 test on the given dataset, using n harmonics and a frequency range defined by highest_freq, lowest_freq, 
    nr_freq. Returns the zn_sq values, frequencies, and pvals. The function uses source_name to get data from the known_clusters.json
    file, but simulated data can be passed though variable toa by setting source_name = false. toa must contain data in units of days.
    """
#    if (not source_name & toa == []):
#        raise ValueError("No source and no data given")
    if highest_freq <= 0:
        raise ValueError("highest_freq must be positive nonzero number")
    freqs = []
    phase_models = []
    zn_sq = []
    toas_d = []
    toas = []
    pvals = []

    # import data
    if source_name:
        with open("D:\Thesis\data\known_clusters.json", "r") as f:
            data = js.load(f)
        for event in data["sources"][source_name]["events"]:
            toas.append(event["timestamp_utc"])
        toas = np.array(toas).astype(np.datetime64)
        toas_d = (toas - toas[0]).astype('timedelta64[s]').astype(float) / 86400 #turn into days from 1st detection
    else:
        source_name = "generated_data"
        toas_d = toa
    #if necessary, flatten 
    if flatten:
        toas_d = toa_flatten(toas_d)
    print(type(toas_d))
    print(toas_d)
    # def freq range
    if not lowest_freq:
        lowest_freq = 1.0 / (toas_d[-1] - toas_d[0])
    if not highest_freq:
        highest_freq = 1.0 / (min(np.diff(toas_d)))
    freqs = np.linspace(lowest_freq, highest_freq , nr_freq)

    # def phase models
    phase_models = np.array([toa_to_phase(i, 0, toas_d) for i in freqs])

    zn_sq = ([zn_squared(i, n) for i in phase_models])
    pvals.append(scipy.stats.chi2.sf(zn_sq, 2*n))

    if plotting:
        # regular plot
        plt.plot(1 / freqs, zn_sq, lw = 0.5, color = "black")
        plt.title(f"$Z_{n}^2$ - {source_name}", fontsize = 15)
        # best period fit
        plt.vlines(1 / freqs[np.argmax(zn_sq)],
                   ymin = min(zn_sq) - 10,
                    ymax = max(zn_sq) + 10,
                      colors = "magenta", linestyles="--",
                        label = f"best period - {1 / freqs[np.argmax(zn_sq)]}",
                        alpha = 0.5)
        # p-value threshold, adjusted for number of trials
        plt.hlines(scipy.stats.chi2.isf(0.001 / nr_freq, 2*n),
                   xmax = 1 / min(freqs),
                   xmin = 1 / max(freqs),
                   colors = "red",
                   label = f"p=0.001 / {nr_freq}",
                   linestyles = "--",
                   alpha = 0.5)
        # mean
        plt.hlines(np.mean(zn_sq),
                   xmax = 1/min(freqs),
                   xmin = 1/max(freqs),
                   colors = "orange",
                   label = f"mean = {np.mean(zn_sq)}",
                   linestyles = "--",
                   alpha = 0.5
                    )
        # median
        plt.hlines(np.median(zn_sq),
            xmax = 1/min(freqs),
            xmin = 1/max(freqs),
            colors = "yellow",
            label = f"median = {np.median(zn_sq)}",
            linestyles = "--",
            alpha = 0.5
            )
        # variance
        plt.hlines(np.var(zn_sq),
            xmax = 1/min(freqs),
            xmin = 1/max(freqs),
            colors = "blue",
            label = f"variance = {np.var(zn_sq)}",
            linestyles = "--",
            alpha = 0.5
            )
        plt.xlabel("period [d]", fontsize = 15)
        plt.ylabel(f"$Z_{n}^2$", fontsize = 15)
        plt.xlim(1/max(freqs), 1/min(freqs))
        plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
        if saving:
            plt.savefig(f"D:\Thesis\\figures\\zsq\\Z_{n}^2_{source_name}", bbox_inches = 'tight')
            plt.show()
        # loglog plot
        plt.loglog(1 / freqs, zn_sq, lw = 0.5, color = "black")
        plt.title(f"$Z_{n}^2$ - {source_name} (loglog)", fontsize = 15)
        plt.vlines(1 / freqs[np.argmax(zn_sq)],
                   ymin = min(zn_sq) - 10,
                    ymax = max(zn_sq) + 10,
                      colors = "magenta",
                        linestyles="--",
                          label = f"best period - {round(1 / freqs[np.argmax(zn_sq)], 3)} [d]")
        plt.hlines(scipy.stats.chi2.isf(0.001 / nr_freq, 2*n),
            xmax = 1 / min(freqs),
            xmin = 1 / max(freqs),
            colors = "red",
            label = f"p=0.001 / {nr_freq}",
            linestyles = "--",
            alpha = 0.5)
        # mean
        plt.hlines(np.mean(zn_sq),
                   xmax = 1/min(freqs),
                   xmin = 1/max(freqs),
                   colors = "orange",
                   label = f"mean = {round(np.mean(zn_sq), 3)}",
                   linestyles = "--",
                   alpha = 0.5
                    )
        # median
        plt.hlines(np.median(zn_sq),
            xmax = 1/min(freqs),
            xmin = 1/max(freqs),
            colors = "yellow",
            label = f"median = {round(np.median(zn_sq), 3)}",
            linestyles = "--",
            alpha = 0.5
            )
        # variance
        plt.hlines(np.var(zn_sq),
            xmax = 1/min(freqs),
            xmin = 1/max(freqs),
            colors = "blue",
            label = f"variance = {round(np.var(zn_sq), 3)}",
            linestyles = "--",
            alpha = 0.5
            )
        plt.xlabel("period [d]", fontsize = 15)
        plt.ylabel(f"$Z_{n}^2$", fontsize = 15)
        plt.xlim(1/max(freqs), 1/min(freqs))
        plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
        if saving:
            plt.savefig(f"D:\Thesis\\figures\\zsq\\Z_{n}^2_{source_name}_loglog", bbox_inches = "tight")
            plt.show()

    return zn_sq, freqs, pvals 