def toa_to_phase(freq, freq_dot, toas):
    """
    Function takes in a frequency and an array of times of arrival (toas) sorted by olderst to newest. Function returns an array made up of
    the corresponding phase value to each date in toas.

    Takes: 
        freq: float, chosen frequency
        freq_dot: float, frequency derivative
        toas: array, times of arrival (in units of time from first observation, usually I use days)

    Returns:
        toas_p: array Phase values corresponding to each toa, calculated according to input frequency and frequency derivative
    """

    t_0 = toas[0]
    toas_p =( freq * (toas - t_0) + freq_dot * (toas - t_0) ** 2 / 2 ) % 1
    return toas_p
