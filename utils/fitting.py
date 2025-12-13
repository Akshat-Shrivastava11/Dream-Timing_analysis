def find_baseline(event_info):
    """Baseline = mean of first 10% of samples."""
    try:
        baseline_region = np.asarray(event_info[: int(0.1 * len(event_info))])
        avg = np.mean(baseline_region)
        rms = np.std(baseline_region)
    except Exception:
        avg = 0
        rms = 99999
    return avg, rms


def fit_region(event_info, baseline):
    """
    Returns rising-edge region (90%→10%) of amplitude, same as original DeltaT.
    """
    try:
        event_info = np.asarray(event_info)
        if event_info.size == 0:
            return np.array([]), np.array([]), 0.0

        min_idx = np.argmin(event_info)
        min_val = event_info[min_idx]
        amplitude = baseline - min_val

        th_low = baseline - 0.1 * amplitude
        th_high = baseline - 0.9 * amplitude

        idxs = np.arange(len(event_info))
        mask = (
            (event_info >= th_high) &
            (event_info <= th_low) &
            (idxs < min_idx) &
            (np.abs(idxs - min_idx) < 10)
        )
        return event_info[mask], idxs[mask], amplitude

    except Exception:
        return np.array([]), np.array([]), 0.0


def fit_rising_edge(region, idxs):
    if len(region) < 2:
        return None, None
    return np.polyfit(idxs, region, 1)


def extract_true_time(slope, intercept, baseline, amplitude):
    """
    === T50 Extraction ===
    Solve: slope * t + intercept = baseline - 0.5 * amplitude
    """
    if slope is None:
        return np.nan

    target = baseline - 0.5 * amplitude
    try:
        t_samp = (target - intercept) / slope  # in samples
    except Exception:
        return np.nan

    return t_samp * SAMPLE_TIME_NS  # convert to ns