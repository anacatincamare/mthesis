import numpy as np
from scipy.stats import weibull_min

def simulate_FRB_repeater_from_weibull(params, n_events=1000, seed=0):
    """
    Function simulates the times of arrival for an FRB repeater by drawing from the Weibull distribution. 
    Times between active periods are sampled from a Weibull distribution.
    The times between bursts within the active period are sampled from another Weibull distribution. 
    """
    # get params, establish starting values
    shape, scale, period, subshape, subscale, subperiod, active_window = params
    rng = np.random.default_rng(seed) #the generator object is used to pull multiple vals from the weibull distribution
    toas = [0.0]
    start = 0.0
    while len(toas) < n_events:
        # detections inside one active window
        t = start
        while True:
            t += weibull_min.rvs(subshape, loc=subperiod, scale=subscale, random_state=rng)
            if t - start > active_window:
                break
            toas.append(t)
        # start of the next active period (measured from the start of this one)
        start += weibull_min.rvs(shape, loc=period, scale=scale, random_state=rng)
        toas.append(start)
    return np.sort(toas)
