import json as js
import matplotlib.pyplot as plt
import numpy as np
import scipy
from astropy import timeseries
import random

from zn2_stat import zn2_stat


source_names = ['FRB20180916B', 'FRB20220912A', 'FRB20190303A', 'FRB20240209A']
data = []
for source in source_names:
    data.append(zn2_stat(source_name=source, n=4, nr_freq= 10000, plotting = True, saving = True))

print(1 / data[0][1][np.argmax(data[0][0])])





