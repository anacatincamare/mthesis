import json as js
import matplotlib.pyplot as plt
import numpy as np
import scipy
from astropy import timeseries
import random
import json
from toa_formatting import toa_flatten

from zn2_stat import zn2_stat


# First Z-stat on the sources
#with open("D:\Thesis\data\known_clusters.json", "r") as f:
#    data = json.load(f)

# find sources with > 50 observations
source_names = ['doubleweibull']
#for source in data["sources"]:
#    if (data["sources"][source]["n_events"] > 50):
#        source_names.append(source)
#source_names = [False]

with open('weibull_test.txt') as fp:
    toas = fp.read()

toas = toas.split('\n')
toas = np.array([float(x) for x in toas[:-1]])

data = [0]
#s = np.random.poisson(100) #sample from poisson w rate 100
#toas_d = np.sort(np.array(np.random.uniform(1, 1500, s)))


for source in source_names:
    data.append(zn2_stat(source_name= False, n=4, lowest_freq = False,  highest_freq = 1/4,
                          nr_freq= 10000, plotting = True, saving = True, flatten = False, toa = toas))

                          
np.savetxt("zn2_weibull.txt", (data[0][1]))





