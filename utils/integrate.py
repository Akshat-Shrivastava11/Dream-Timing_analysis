def integrate_waveform(events, window=100, baseline_samples=20):
    integrals = []
    for event in events:
        wf = np.array(event)
        baseline = np.mean(wf[:baseline_samples])
        corr = wf - baseline
        peak = np.argmin(corr)

        start = max(0, peak - window)
        end   = min(len(corr), peak + window)

        area = np.trapezoid(corr[start:end], dx=1)
        integrals.append(-area)    # positive area
    return np.array(integrals)