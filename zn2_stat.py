def zn2_stat(source_name, n, nr_freq = 5000, highest_freq = 0.5, plotting = False, saving = False, toa = []):
    """
    The function does the zn^2 test on the given dataset, using n harmonics and a frequency range defined by highest_freq, lowest_freq, 
    nr_freq. Returns the zn_sq values, frequencies, and pvals. The function uses source_name to get data from the known_clusters.json
    file, but simulated data can be passed though variable toa by setting source_name = false. toa must contain data in units of days.
    """
    if (not source_name & toa == []):
        raise ValueError("No source and no data given")
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
            data = json.load(f)
        for event in data["sources"][source]["events"]:
            toas.append(event["timestamp_utc"])
        toas = np.array(toas).astype(np.datetime64)
        toas_d = (toas - toas[0]).astype('timedelta64[s]').astype(float) / 86400 #turn into days from 1st detection
    else:
        source_name = "no name"
        toas_d = toa

    # def freq range
    lowest_freq = 1.0 / (toas_d[-1] - toas_d[0])
    print(highest_freq, lowest_freq)
    freqs = np.linspace(lowest_freq, highest_freq , nr_freq)

    # def phase models
    phase_models = np.array([phase(i, 0, toas_d) for i in freqs])

    zn_sq = ([zn_squared(n, i) for i in phase_models])
    pvals.append(scipy.stats.chi2.sf(zn_sq, 2*n))

    if plotting:
        # regular plot
        plt.plot(1 / freqs, zn_sq, lw = 0.5, color = "black")
        plt.title(f"$Z_{n}^2$ - {source_name}", fontsize = 15)
        plt.xlabel("period [d]", fontsize = 15)
        plt.ylabel(f"$Z_{n}^2$", fontsize = 15)
        if saving:
            plt.savefig(f"D:\Thesis\figures\zsq\Z_{n}^2_{source_name}")
        # loglog plot
        plt.loglog(1 / freqs, zn_sq, lw = 0.5, color = "black")
        plt.title(f"$Z_{n}^2$ - {source_name} (loglog)", fontsize = 15)
        plt.xlabel("period [d]", fontsize = 15)
        plt.ylabel(f"$Z_{n}^2$", fontsize = 15)
        if saving:
            plt.savefig(f"D:\Thesis\figures\zsq\Z_{n}^2_{source_name}_loglog")

    return zn_sq, freqs, pvals 