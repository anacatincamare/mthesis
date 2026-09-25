
def zn_squared(phase_model, n):
    """
    Function takes the phase model and the number of harmonics (n) and returns the Zn-squared value of that phase model.

    Takes: 
        phase_model: array of times of arrivals converted into phase model.
        n: int, number of harmonics
    """
    N = len(toas_d) # total number of detections
    phases = 2*np.pi*np.outer(np.arange(1, n+1), toas_d)
    zn_sq = (2.0 / N) * np.sum(np.sum(np.cos(phases), axis = 1) ** 2 + np.sum(np.sin(phases), axis = 1) ** 2)
    return zn_sq
