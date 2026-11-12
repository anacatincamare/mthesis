import numpy as np
import math

# want inhom poisson, where the rate follows a sinusoidal (or any other given) function. i need the function to know where it is on the x-axis at all times.
# could do this by having a function that gets called every time the inhom poisson needs a new rate (every day?) 
# define the rate parameter over the entire elapsed window! then have a separate toa generation for the inhom poisson??


def inhom_poisson_window(window_start, rates, window, rng):
    toas = [np.floor(window_start), np.floor(window_start + window)]
    print()
    for rate in rates[np.floor(window_start):np.floor(window_start + window)]:
        toas.append(poisson_window(rate, window = 1, rng = rng))
    return toas
    


def poisson_window(rate, window, rng):
    n = rng.poisson(rate * window) #draw number of events in window. rate is given at nr / day, so it is adjusted by active window duration
    return np.sort(rng.uniform(0.0, window, n)) #generate events according to poisson


def weibull_widndow(shape, mean_wait, window, rng):
    """
    Function creates toas in the given active window by drawing wait times between toas from a Weibull distribution with a given shape and mean wait. The rng is a random state seed.

    Args:
        shape: np.float - shape parameter for Weibull distribution
        mean_wait: np.float - [d], gets reparametrized to the scale parameter
        window: np.float - [d], elapsed time since start of active window 
        rng: random generator. derived from the seed given in main function
    
    Returns:
        toas: array-like - times of arrival [d since window start]
    """
    out, t = [], 0.0
    scale = mean_wait / math.gamma(1.0 + 1.0 / shape)
    while True:
        c = t + np.cumsum(scale * rng.weibull(shape, 64))
        inside = c < window
        out.append(c[inside])
        if not inside.all():   # a draw crossed the window end -> done
            break
        t = c[-1]
    toas =  np.concatenate(out)
    return toas

def generate_repeater_from_dist(
    elapsed_time,
    period = False,
    window = False,
    dist="poisson",
    rate=None,          # Poisson: events per day of active time
    shape=None,         # Weibull shape k
    mean_wait=None,         # Weibull mean wait (d) (reparametrizes into scale)
    rates = None, # Expected rates for inhomogeneous Poisson distribution. Must be defined over the elapsed time
    window_start=0.0,   # offset (days) of the window start within the period. if = 0, the active window starts with the first observation
    t0=0.0,             # first period start
    seed=None,          # reccommended to set a seed for reproductability
    max_windows=10_000_000,  # safety, in case emission late is too low and n_points is unreasonable to reach
):
    """
    Simulates an array with times of arrival (days). The source is observable only during
    windows of length `window`, starting every `period` days. This method assumes 
    the source emits every period and continuously during the active window. If no period is given,
    the simulation generates n_points within one continuous time segment. This method
    assumes that the period and active windows are strict (same duration every time).
    Only Poisson and Weibull are available for now.

    Poisson : per window, draw N ~ Poisson(rate * window in days), then place the N
              events uniformly in the window, following the dist of interarrival times for Poisson
    Weibull : per window, restart a renewal process at the window start, with
              Weibull-distributed waiting times; events beyond the window end
              are discarded.

    Returns a sorted 1-D array of length n_points.
    """
    rng = np.random.default_rng(seed)
    dist = dist.lower() #lowercasing
    if not period:
        window = elapsed_time

    # check that dist are available - def events method
    if dist == "poisson":
        if rate is None or rate <= 0:
            raise ValueError("Poisson requires rate > 0")
        def events_in_one_window(window_start):
            return poisson_window(rate, window, rng)
    elif dist == "weibull":
        if shape is None or mean_wait is None or shape <= 0 or mean_wait <= 0:
            raise ValueError("Weibull requires shape > 0 and mean_wait > 0")
        def events_in_one_window(window_start):
            return  weibull_widndow(shape, mean_wait, window, rng)
    elif dist == "inhomogeneous_poisson":
        if len(rates) < elapsed_time:
            raise ValueError ("Rate array must be defined over entire elapsed period")
        def events_in_one_window(window_start):
            return inhom_poisson_window(window_start, rates, window, rng)
        
    else:
        raise ValueError("dist must be 'poisson', 'weibull', or 'inhomogeneous_poisson'")
    

    # fill windows
    origin = t0 + window_start
    toas, last_obs = [], 0
    if not period:
        max_windows = 1

    for k in range(max_windows):
        ev = events_in_one_window(window_start = origin + k * period)
        if ev.size:
            toas.append(origin + k * period + ev)
            last_obs = toas[-1][-1]
        if last_obs >= elapsed_time:
            break
#    else:
#        raise RuntimeError("max_windows reached before collecting n_points")
    toas = np.concatenate(toas)   # drop the excess (latest) events
    toas = np.array(toas)
    mask = toas < elapsed_time
    return toas[mask]


def observing_window_mask(toas, window = 10 / 1440, period = 1.0, t_offset = 0.0):
    """
    Masks the toas not included in a certain observing window. The observing window default is set to the CHIME obserwing window of 10 minutes once every 24 hours at midnight.
    Function should be used on generated data, as real data is already automatically "masked".

    Returns mask that excludes unobserved bursts.
    """
#    t_offset = toas[0]
    toas_rem = np.mod(toas - t_offset, period)
    return toas_rem < window
