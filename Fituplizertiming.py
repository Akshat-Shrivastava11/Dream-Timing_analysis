#!/usr/bin/env python3
"""
TimingDAQ-like Python ntuplizer:
  - Minimal (default): writes per-channel
        <channel>_t_peak
        <channel>_LP2_50
  - If --all_features: additionally writes per-channel
        <channel>_gaus_mean
        <channel>_gaus_sigma
        <channel>_gaus_chi2
        <channel>_linear_RE_<NN>        for NN = int(100*f) over config.constant_fraction
        <channel>_linear_RE__<TT>mV     for TT = int(abs(thr)) over config.constant_threshold

Logic is matched to DRSAnalyzer.cc for:
  - baseline windowing (baseline_time fractions)
  - shift hack for i%9==8 and i<=287 (150 samples, +30 ns)
  - peak picking logic (idx_min, amp)
  - Gaussian window selection using GetIdxFirstCross(amp*frac, ...)
  - Rising-edge window selection using Re bounds and GetIdxFirstCross
  - linear time extraction formulas and myTimeOffset usage

Dependencies:
  - numpy, uproot, awkward
  - scipy (optional): used for Gaussian curve_fit if available; falls back to moment-based estimate otherwise
"""

import argparse
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import uproot
import awkward as ak

# Optional scipy for gaussian fit
try:
    from scipy.optimize import curve_fit  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    curve_fit = None
    _HAVE_SCIPY = False

# ---------------------------
# Constants matching C++
# ---------------------------
NUM_SAMPLES = 900
DT_NS = 0.2  # ns per sample, matches C++: (200.0/1000.0)*i
CHANNEL_RE = re.compile(r"^DRS_Board\d+_Group\d+_Channel\d+$")

# C++ shift hack: i == 8,17,26,...,287  (i%9==8 and i<=287)
SHIFT_MOD = 9
SHIFT_REM = 8
SHIFT_MAX_I = 287
SHIFT_SAMPLES = 150


def is_channel_branch(name: str) -> bool:
    return CHANNEL_RE.match(name) is not None


def ch_out(chname: str, var: str) -> str:
    return f"{chname}_{var}"


# ---------------------------
# Config structures
# ---------------------------
@dataclass
class ChannelCfg:
    N: int
    baseline_time: Tuple[float, float] = (0.0, 0.0)  # fractions of NUM_SAMPLES
    polarity: int = +1
    counter_auto_pol_switch: int = -1
    amplification: float = 0.0
    attenuation: float = 0.0
    algorithm: str = ""
    gaus_fraction: float = 0.4
    re_bounds: Tuple[float, float] = (0.15, 0.75)
    PL_deg: List[int] = field(default_factory=list)
    weierstrass_filter_width: float = 0.0


@dataclass
class Config:
    channels: Dict[int, ChannelCfg] = field(default_factory=dict)
    constant_fraction: List[float] = field(default_factory=lambda: [0.15, 0.3, 0.45])
    constant_threshold: List[float] = field(default_factory=list)
    verbose: bool = False

    def has_channel(self, i: int) -> bool:
        return i in self.channels

    def get_mult_factor(self, i: int) -> float:
        """
        Conventional TimingDAQ multiplier:
          polarity * 10^(amplification/20) * 10^(-attenuation/20)

        If your src/Configuration.cc differs, copy it exactly here.
        """
        c = self.channels[i]
        pol = float(c.polarity)
        gain = 10.0 ** (c.amplification / 20.0)
        att = 10.0 ** (-c.attenuation / 20.0)
        return pol * gain * att


# ---------------------------
# Config parser (AUTO baseline)
# ---------------------------
def _parse_baseline_tokens_auto(a: float, b: float) -> Tuple[float, float]:
    """
    Supports both common TimingDAQ formats:
      - baseline <ch> <start_frac> <end_frac>   (if end<=1.0)
      - baseline <ch> <start_idx>  <n_samples>  (if end>1.0)
    """
    if b <= 1.0:
        st_frac = float(a)
        en_frac = float(b)
    else:
        start_idx = float(a)
        n_samp = float(b)
        st_frac = start_idx / float(NUM_SAMPLES)
        en_frac = (start_idx + n_samp) / float(NUM_SAMPLES)

    st_frac = max(0.0, min(st_frac, 1.0))
    en_frac = max(0.0, min(en_frac, 1.0))
    if en_frac <= st_frac:
        en_frac = min(1.0, st_frac + (1.0 / float(NUM_SAMPLES)))
    return (st_frac, en_frac)


def parse_config_file(path: str, verbose: bool = False) -> Config:
    cfg = Config(verbose=verbose)

    re_rebounds = re.compile(r"Re(\d\d)-(\d\d)")
    re_gfrac = re.compile(r"G(\d\d)")

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if (not line) or line.startswith("#"):
                continue

            toks = line.split()
            if not toks:
                continue

            key0 = toks[0].lower()

            # Global settings
            if key0 == "constant_fraction":
                vals = []
                for x in toks[1:]:
                    try:
                        vals.append(0.01 * float(x))  # percent -> fraction (C++)
                    except Exception:
                        pass
                if vals:
                    cfg.constant_fraction = vals
                continue

            if key0 == "constant_threshold":
                vals = []
                for x in toks[1:]:
                    try:
                        vals.append(float(x))
                    except Exception:
                        pass
                cfg.constant_threshold = vals
                continue

            # Baseline line
            if key0 == "baseline" and len(toks) >= 4:
                try:
                    ch = int(toks[1])
                    a = float(toks[2])
                    b = float(toks[3])
                except Exception:
                    continue
                if ch not in cfg.channels:
                    cfg.channels[ch] = ChannelCfg(N=ch)
                cfg.channels[ch].baseline_time = _parse_baseline_tokens_auto(a, b)
                continue

            # Per-channel line:
            # CH  POLARITY  AMPLIFICATION  ATTENUATION  ALGORITHM  FILTER_WIDTH
            if toks[0].isdigit() and len(toks) >= 6:
                ch = int(toks[0])

                pol_tok = toks[1]
                if pol_tok == "+":
                    polarity = +1
                elif pol_tok == "-":
                    polarity = -1
                else:
                    polarity = int(pol_tok)

                amplification = float(toks[2])
                attenuation = float(toks[3])
                algorithm = toks[4]
                wfilt = float(toks[5])

                c = ChannelCfg(
                    N=ch,
                    polarity=polarity,
                    amplification=amplification,
                    attenuation=attenuation,
                    algorithm=algorithm,
                    weierstrass_filter_width=wfilt,
                )

                # Extract Re bounds if present: Re##-##
                m = re_rebounds.search(algorithm)
                if m:
                    c.re_bounds = (int(m.group(1)) / 100.0, int(m.group(2)) / 100.0)

                # Extract G fraction if present: G##
                m = re_gfrac.search(algorithm)
                if m:
                    c.gaus_fraction = int(m.group(1)) / 100.0

                # Extract LP degrees (not used here beyond LP2_50)
                for deg in (1, 2, 3):
                    if f"LP{deg}" in algorithm:
                        c.PL_deg.append(deg)

                cfg.channels[ch] = c
                continue

            # Otherwise ignore
    return cfg


# ---------------------------
# Core algorithms matching C++
# ---------------------------
def get_idx_first_cross(value: float, v: np.ndarray, i_st: int, direction: int) -> int:
    """
    Exact logic from C++ GetIdxFirstCross()
    """
    idx_end = (len(v) - 1) if direction > 0 else 0
    rising = (value > v[i_st])
    i = int(i_st)

    while i != idx_end:
        if rising and (v[i] > value):
            break
        if (not rising) and (v[i] < value):
            break
        i += int(direction)
        if i < 0:
            i = 0
            break
        if i >= len(v):
            i = len(v) - 1
            break
    return int(i)


def poly_fit_time_as_func_of_amp(x_amp: np.ndarray, y_time: np.ndarray, deg: int) -> np.ndarray:
    X = np.vstack([x_amp**k for k in range(deg + 1)]).T
    coeff, *_ = np.linalg.lstsq(X, y_time, rcond=None)
    return coeff.astype(float)


def poly_eval(x: float, coeff: np.ndarray) -> float:
    out = 0.0
    for k, c in enumerate(coeff):
        out += c * (x**k)
    return float(out)


# ---------------------------
# Gaussian fit (C++-like window, python fit)
# ---------------------------
def _gaus(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_gaussian_peak(t: np.ndarray, y: np.ndarray, j_down: int, j_up: int, amp: float) -> Tuple[float, float, float]:
    """
    Mimics C++:
      TF1 gaus in [time[j_down], time[j_up]], fit pulse, then:
        gaus_mean = mu (+ myTimeOffset applied outside)
        gaus_sigma = sigma
        gaus_chi2 = pulse->Chisquare(fpeak,"R")
    Here:
      - if scipy available: curve_fit on the window
      - else: moment-based mu/sigma and SSE as chi2-like
    """
    if j_up <= j_down:
        return 0.0, 0.0, 0.0

    lo = max(0, j_down)
    hi = min(len(t) - 1, j_up)
    if hi - lo + 1 < 5:
        return 0.0, 0.0, 0.0

    x = t[lo:hi + 1]
    yy = y[lo:hi + 1]

    # initial guesses inspired by C++ intent
    mu0 = float(x[np.argmin(yy)])  # negative peak typically
    sigma0 = float(max(x[-1] - x[0], 1e-3))
    A0 = float(amp * np.sqrt(2.0 * np.pi) * sigma0)

    if _HAVE_SCIPY:
        try:
            popt, _ = curve_fit(_gaus, x, yy, p0=[A0, mu0, sigma0], maxfev=4000)
            yhat = _gaus(x, *popt)
            chi2 = float(np.sum((yy - yhat) ** 2))
            mu = float(popt[1])
            sig = float(abs(popt[2]))
            return mu, sig, chi2
        except Exception:
            pass

    # Fallback: estimate mu/sigma by weighting with (-yy) (since peak is negative)
    w = np.clip(-yy, 0.0, None)
    if float(np.sum(w)) <= 0.0:
        return 0.0, 0.0, 0.0
    mu = float(np.sum(w * x) / np.sum(w))
    sig = float(np.sqrt(np.sum(w * (x - mu) ** 2) / np.sum(w)))
    yhat = _gaus(x, A0, mu, max(sig, 1e-6))
    chi2 = float(np.sum((yy - yhat) ** 2))
    return mu, sig, chi2


# ---------------------------
# Per-event feature computation
# ---------------------------
def compute_features(
    wf_in: np.ndarray,
    time: np.ndarray,
    cfg: Config,
    ch_idx: int,
    all_features: bool,
) -> Dict[str, float]:
    """
    Always returns:
      t_peak, LP2_50  (LP2_50 is 0 if algorithm doesn't include LP2 or not fittable)
    If all_features=True:
      gaus_mean, gaus_sigma, gaus_chi2 (0 if algorithm doesn't include G)
      linear_RE_* and linear_RE__*mV (0 if algorithm doesn't include Re)
    """
    c = cfg.channels[ch_idx]
    algo = c.algorithm

    out: Dict[str, float] = {}
    # Defaults (C++ ResetVar sets all computed vars to 0 each event)
    # We'll only write branches requested by mode; caller will fill missing with zeros anyway.

    # Baseline indices (C++ uses baseline_time fractions)
    bl_st = int(c.baseline_time[0] * NUM_SAMPLES)
    bl_en = int(c.baseline_time[1] * NUM_SAMPLES)
    bl_st = max(0, min(bl_st, NUM_SAMPLES - 1))
    bl_en = max(bl_st + 1, min(bl_en, NUM_SAMPLES))
    bl_len = bl_en - bl_st

    baseline = float(np.mean(wf_in[bl_st:bl_en]))
    scale_factor = float(cfg.get_mult_factor(ch_idx))

    # Shift hack (same as C++)
    myTimeOffset = 0.0
    wf = wf_in.astype(float, copy=True)
    if (ch_idx % SHIFT_MOD == SHIFT_REM) and (ch_idx <= SHIFT_MAX_I):
        wf_new = np.full(NUM_SAMPLES, baseline, dtype=float)
        if SHIFT_SAMPLES < wf.shape[0]:
            copy_len = min(wf.shape[0] - SHIFT_SAMPLES, NUM_SAMPLES)
            wf_new[:copy_len] = wf[SHIFT_SAMPLES:SHIFT_SAMPLES + copy_len]
        wf = wf_new
        myTimeOffset = SHIFT_SAMPLES * DT_NS

    # Baseline subtract + scale (no HNR in this python version)
    wf = scale_factor * (wf - baseline)

    # Peak picking exactly like C++
    amp = 0.0
    idx_min = bl_en

    for j in range(NUM_SAMPLES):
        range_check = (j > (bl_st + bl_len)) and (j < NUM_SAMPLES)
        if c.counter_auto_pol_switch > 0:
            max_check = abs(wf[j]) > abs(amp)
        else:
            max_check = wf[j] < amp
        if (range_check and max_check) or (j == (bl_st + bl_len)):
            idx_min = j
            amp = wf[j]

    out["t_peak"] = float(time[idx_min]) if 0 <= idx_min < NUM_SAMPLES else 0.0

    # baseline_RMS as computed in C++ after subtraction/scaling
    baseline_RMS = float(np.sqrt(np.mean(wf[bl_st:bl_en] ** 2))) if bl_len > 0 else 0.0

    # fittable conditions (same as C++)
    fittable = True
    fittable &= (idx_min < int(NUM_SAMPLES * 0.8))
    fittable &= (abs(amp) > 5.0 * baseline_RMS)
    if 2 <= idx_min <= (NUM_SAMPLES - 3):
        fittable &= (abs(wf[idx_min + 1]) > 2.0 * baseline_RMS)
        fittable &= (abs(wf[idx_min - 1]) > 2.0 * baseline_RMS)
        fittable &= (abs(wf[idx_min + 2]) > 1.0 * baseline_RMS)
        fittable &= (abs(wf[idx_min - 2]) > 1.0 * baseline_RMS)
    else:
        fittable = False

    # ---------------------------
    # LP2_50 (t50) — only if minimal or all_features AND algo contains LP2
    # ---------------------------
    out["LP2_50"] = 0.0
    if fittable and ("None" not in algo) and ("LP2" in algo):
        f = 0.50
        deg = 2

        j_10_pre = get_idx_first_cross(amp * 0.1, wf, idx_min, -1)
        j_90_pre = get_idx_first_cross(amp * 0.9, wf, j_10_pre, +1)

        start_level = -3.0 * baseline_RMS
        j_start = get_idx_first_cross(start_level, wf, idx_min, -1)

        j_st = j_start
        if (amp * f) > start_level:
            j_st = get_idx_first_cross(amp * f, wf, idx_min, -1)

        j_close = get_idx_first_cross(amp * f, wf, j_st, +1)
        if (j_close - 1) >= 0 and abs(wf[j_close - 1] - f * amp) < abs(wf[j_close] - f * amp):
            j_close -= 1

        span_j = int(min(j_90_pre - j_close, j_close - j_st) / 1.5)
        if (j_90_pre - j_10_pre) <= 3 * deg:
            span_j = max(int(deg * 0.5), span_j)
            span_j = max(1, span_j)
        else:
            span_j = max(deg, span_j)

        if (j_close >= span_j) and (j_close + span_j < NUM_SAMPLES):
            N_add = 1
            if (span_j + N_add + j_close) < j_90_pre:
                N_add += 1
            lo = j_close - span_j
            hi = j_close + span_j + N_add  # exclusive

            if 0 <= lo < hi <= NUM_SAMPLES:
                coeff = poly_fit_time_as_func_of_amp(wf[lo:hi], time[lo:hi], deg=deg)
                out["LP2_50"] = poly_eval(f * amp, coeff) + myTimeOffset

    if not all_features:
        return out

    # ---------------------------
    # Gaussian fit features (if algo contains "G")
    # ---------------------------
    out["gaus_mean"] = 0.0
    out["gaus_sigma"] = 0.0
    out["gaus_chi2"] = 0.0

    if fittable and ("G" in algo):
        frac = float(c.gaus_fraction)
        j_down = get_idx_first_cross(amp * frac, wf, idx_min, -1)
        j_up = get_idx_first_cross(amp * frac, wf, idx_min, +1)
        if (j_up - j_down) < 4:
            j_up = min(NUM_SAMPLES - 2, idx_min + 1)
            j_down = max(1, idx_min - 1)

        mu, sig, chi2 = fit_gaussian_peak(time, wf, j_down, j_up, amp)
        out["gaus_mean"] = float(mu + myTimeOffset)  # C++ adds myTimeOffset here
        out["gaus_sigma"] = float(sig)
        out["gaus_chi2"] = float(chi2)

    # ---------------------------
    # Linear rising-edge features (if algo contains "Re")
    # ---------------------------
    # Names must match C++:
    #   linear_RE_%d  (int(100*f))
    #   linear_RE__%dmV (double underscore, abs(thr))
    if "Re" in algo and fittable:
        rb0, rb1 = c.re_bounds

        i_min = get_idx_first_cross(rb0 * amp, wf, idx_min, -1)
        i_max = get_idx_first_cross(rb1 * amp, wf, i_min, +1)

        # Guard
        if 0 <= i_min < i_max < NUM_SAMPLES:
            t_min = float(time[i_min])
            t_max = float(time[i_max])

            # Fit y = m*t + b over [t_min, t_max] using points in that index range
            # (In C++ they do pulse->Fit("flinear", "R...") which is least squares)
            x = time[i_min:i_max + 1].astype(float)
            y = wf[i_min:i_max + 1].astype(float)

            if x.size >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
                # polyfit returns [m, b]
                m, b = np.polyfit(x, y, 1)
                if m != 0.0:
                    for ffrac in cfg.constant_fraction:
                        nn = int(round(100.0 * ffrac))
                        out[f"linear_RE_{nn}"] = float((ffrac * amp - b) / m + myTimeOffset)
                    for thr in cfg.constant_threshold:
                        tt = int(round(abs(thr)))
                        out[f"linear_RE__{tt}mV"] = float((thr - b) / m + myTimeOffset)
                else:
                    # keep zeros
                    for ffrac in cfg.constant_fraction:
                        nn = int(round(100.0 * ffrac))
                        out[f"linear_RE_{nn}"] = 0.0
                    for thr in cfg.constant_threshold:
                        tt = int(round(abs(thr)))
                        out[f"linear_RE__{tt}mV"] = 0.0
            else:
                for ffrac in cfg.constant_fraction:
                    nn = int(round(100.0 * ffrac))
                    out[f"linear_RE_{nn}"] = 0.0
                for thr in cfg.constant_threshold:
                    tt = int(round(abs(thr)))
                    out[f"linear_RE__{tt}mV"] = 0.0
        else:
            for ffrac in cfg.constant_fraction:
                nn = int(round(100.0 * ffrac))
                out[f"linear_RE_{nn}"] = 0.0
            for thr in cfg.constant_threshold:
                tt = int(round(abs(thr)))
                out[f"linear_RE__{tt}mV"] = 0.0
    else:
        for ffrac in cfg.constant_fraction:
            nn = int(round(100.0 * ffrac))
            out[f"linear_RE_{nn}"] = 0.0
        for thr in cfg.constant_threshold:
            tt = int(round(abs(thr)))
            out[f"linear_RE__{tt}mV"] = 0.0

    return out

# ---------------------------
# Main
# ---------------------------
def main():
    import os
    ap = argparse.ArgumentParser(
        description="TimingDAQ-like ntuplizer in Python (t_peak + LP2_50; add gaus/linear with --all_features)."
    )
    ap.add_argument("-i", "--input", required=True, help="Input ROOT file")
    ap.add_argument("-o", "--output", required=True,
                    help="Output ROOT filename (no path, use --outdir)")
    ap.add_argument("--tree", default="EventTree", help="Tree name (default: EventTree)")
    ap.add_argument(
        "--config",
        default="/lustre/research/hep/cmadrid/TimingDAQ/DRS_Service_TTU_2025_Sep.config",
        help="TimingDAQ config file path (default hardcoded).",
    )
    ap.add_argument("--outdir", default="/PostTimingFitskims", help="Output directory (default: .)")
    ap.add_argument("--tag", default="", help="Tag appended to output filename (before .root)")
    ap.add_argument("--all_features", action="store_true",
                    help="Also compute gaus_* and linear_RE_* branches.")
    ap.add_argument("--chunk", type=int, default=2000,
                    help="Events per chunk (default: 2000)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--copy_nonchannel", action="store_true",
                    help="Copy non-channel branches to output")
    ap.add_argument("--copy_waveforms", action="store_true",
                    help="Copy waveform vector branches to output (large)")
    args = ap.parse_args()

    # ---------------------------
    # Load config
    # ---------------------------
    cfg = parse_config_file(args.config, verbose=args.verbose)

    # time array (same as C++)
    time = (DT_NS * np.arange(NUM_SAMPLES)).astype(np.float32)

    # ---------------------------
    # Read input & process
    # ---------------------------
    with uproot.open(args.input) as f:
        tree = f[args.tree]
        keys = list(tree.keys())

        channel_branches = sorted([k for k in keys if is_channel_branch(k)])
        if not channel_branches:
            raise RuntimeError("No DRS_Board*_Group*_Channel* waveform branches found in tree.")

        non_channel = [k for k in keys if (k not in channel_branches and k != "time")]

        read_branches = list(channel_branches)
        if args.copy_nonchannel:
            read_branches += non_channel

        out_parts: Dict[str, List[np.ndarray]] = {}

        def append_out(name: str, arr):
            out_parts.setdefault(name, []).append(arr)

        # Derived suffixes
        derived_suffixes = ["t_peak", "LP2_50"]
        if args.all_features:
            derived_suffixes += ["gaus_mean", "gaus_sigma", "gaus_chi2"]
            for ffrac in cfg.constant_fraction:
                nn = int(round(100.0 * ffrac))
                derived_suffixes.append(f"linear_RE_{nn}")
            for thr in cfg.constant_threshold:
                tt = int(round(abs(thr)))
                derived_suffixes.append(f"linear_RE__{tt}mV")

        # Loop in chunks
        for arrays in tree.iterate(read_branches, step_size=args.chunk, library="ak"):
            n_evt = len(arrays[read_branches[0]])

            if args.copy_nonchannel:
                for k in non_channel:
                    if k in arrays:
                        append_out(k, np.asarray(arrays[k]))

            if args.copy_waveforms:
                for chname in channel_branches:
                    append_out(chname, ak.to_list(arrays[chname]))

            for i, chname in enumerate(channel_branches):
                chunk_out = {suf: np.zeros(n_evt, dtype=np.float32)
                             for suf in derived_suffixes}

                if not cfg.has_channel(i):
                    for suf in derived_suffixes:
                        append_out(ch_out(chname, suf), chunk_out[suf])
                    continue

                wf_jag = arrays[chname]
                wf_fixed = ak.pad_none(wf_jag[:, :NUM_SAMPLES],
                                       NUM_SAMPLES, clip=True)
                wf_np = ak.to_numpy(wf_fixed)
                wf_np = np.asarray(wf_np)
                if np.ma.isMaskedArray(wf_np):
                    wf_np = wf_np.filled(np.nan)
                wf_np = wf_np.astype(np.float32, copy=False)

                for e in range(n_evt):
                    wf = wf_np[e]
                    if not np.isfinite(wf).any():
                        continue
                    finite = np.isfinite(wf)
                    if not finite.all():
                        last = float(wf[finite][-1]) if finite.any() else 0.0
                        wf = np.where(finite, wf, last)

                    feats = compute_features(
                        wf_in=wf,
                        time=time,
                        cfg=cfg,
                        ch_idx=i,
                        all_features=args.all_features,
                    )
                    for suf in derived_suffixes:
                        chunk_out[suf][e] = float(feats.get(suf, 0.0))

                for suf in derived_suffixes:
                    append_out(ch_out(chname, suf), chunk_out[suf])

            append_out("time", np.tile(time, (n_evt, 1)))

    # ---------------------------
    # Concatenate
    # ---------------------------
    final = {}
    for k, parts in out_parts.items():
        if args.copy_waveforms and is_channel_branch(k):
            final[k] = ak.Array(sum(parts, []))
        else:
            final[k] = np.concatenate(parts, axis=0)

    # ---------------------------
    # Build output path
    # ---------------------------
    outname = args.output  # filename only
    if args.tag:
        base, ext = os.path.splitext(outname)
        outname = f"{base}_{args.tag}{ext or '.root'}"

    outpath = os.path.join(args.outdir, outname)
    os.makedirs(args.outdir, exist_ok=True)

    # ---------------------------
    # Write once
    # ---------------------------
    with uproot.recreate(outpath) as fout:
        fout[args.tree] = final

    print(f"Wrote: {outpath}")
    print(f"Tree : {args.tree}")
    print(f"--all_features: {args.all_features}")
    print(f"Branches written: {len(final)}")
