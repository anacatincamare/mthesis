import numpy as np
from scipy.stats import weibull_min, ks_2samp
from scipy.optimize import differential_evolution
from toa_formatting import import_toas, toa_to_days_from_first_detection, toa_flatten, toa_to_phase
from old.simulate_FRB_repeater_from_weibull import simulate_FRB_repeater_from_weibull

#from import_toas import import_toas

#import objective


def fit_2w_model_to_repeater(source_name, bounds, seed, maxiter, n_events = 1000, sim_seed = 0 ):
    #first get flattened toas from source

    #import
    toas = import_toas(source_name)
    toas = toa_to_days_from_first_detection(toas)
    toas = toa_flatten(toas)
    obs_gaps = np.diff(np.sort(toas))

    def objective(p):
        sim = toa_flatten(simulate_FRB_repeater_from_weibull(p))
        gaps = np.diff(sim)
        return ks_2samp(np.log(gaps), np.log(obs_gaps)).statistic

    #perform fitting
    res = differential_evolution(objective, bounds=bounds, seed=seed, maxiter=maxiter, polish=False)
    best_toas = toa_flatten(simulate_FRB_repeater_from_weibull(res.x, n_events=n_events, seed=sim_seed))
    print(res.x, res.fun)

    return (res.x, res.fun, best_toas)

# shape, scale, period, subshape, subscale, subperiod, active_window
bounds = [(0.5, 5), (0.5, 2), (1, 20),
          (0.5, 5), (0.1, 1), (0.5, 10),
          (1, 15)]

params, f, best_toas = fit_2w_model_to_repeater("FRB20180916B", bounds, seed = 42, maxiter = 300, sim_seed = 42)
np.savetxt("best_r3_2w_toas.txt", best_toas)

print(params, f)
