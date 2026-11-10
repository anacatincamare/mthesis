# draw from weibull
import scipy.stats
import numpy as np

def draw_from_weibull(shape, scale, period, nr_draws = 100):
    """
    Function generates a periodic signal in the form of a list of times of arrival (toas) by drawing from a Weibull distribution.

     In: 
        shape: float - shape parameter for weibull distribution
        scale: float - scale parameter for weibull distribution
        period: float -  desired period of the signal, used as the mean of the weibull distribution when drawing toas
        nr_draws: int - number of toas desired
    Out: 
        vars: list - distances between consecutive datapoints
        toas: list - times of arrival, starting at 0 
    """
    # set up data
    toas = np.array(0)
    date = 0
    vars = scipy.stats.weibull_min.rvs(c = shape,
                                        loc = 16 - scipy.stats.weibull_min.mean(c = shape, scale = scale),
                                        scale = scale, 
                                        size = nr_draws)

    for v in vars:
        date = date + v
        toas = np.append(toas, date)

    return toas