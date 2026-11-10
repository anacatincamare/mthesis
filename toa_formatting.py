import json as js
import numpy as np

def import_toas(source_name):
    """
    Function imports the times of arrival in datetime format from the known_clusters dataset by using the name of the source.
    """
    toas = np.array([])
    with open("D:\Thesis\data\known_clusters.json", "r") as f:
        data = js.load(f)
    for event in data["sources"][source_name]["events"]:
        toas = np.append(toas, event["timestamp_utc"])
    return toas

def toa_flatten(toas):
    """
    Function takes an array of times of arrival (in terms of days), and returns an array where
    the toas are integers. Duplicates are merged (if there are 2 or more observations within
    the same integer day, there will only be one array item corresponding to that day).
    """
    return np.unique(np.round(toas))   # integer days, duplicates merged

def toa_to_days_from_first_detection(toas):
    """
    Takes times of arrival as expressed in the known_clusters data and returns an array with the same data in units of days from first detection, with the first detection being always day 1
    """

    toas = np.array(toas).astype(np.datetime64)
    toas_d = (toas - toas[0]).astype('timedelta64[s]').astype(float) / 86400 #turn into days from 1st detection

    return (toas_d)

def toa_to_phase(freq, freq_dot, toas):
    """
    Function takes in a frequency and an array of times of arrival (toas) sorted by olderst to newest. Function returns an array made up of
    the corresponding phase value to each date in toas.

    Takes: 
        freq: float, chosen frequency
        freq_dot: float, frequency derivative
        toas: array, times of arrival (in units of time from first observation, usually I use days)

    Returns:
        toas_p: array, Phase values corresponding to each toa (phase model), calculated according to input frequency and frequency derivative.
        
    """

    t_0 = toas[0]
    toas_p =( freq * (toas - t_0) + freq_dot * (toas - t_0) ** 2 / 2 ) % 1
    return toas_p
