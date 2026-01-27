#!/usr/bin/env python3
"""
TimingDAQ-like Python ntuplizer (multiprocessing, chunked IO) with logic matched to the provided C++:

Matches DRSAnalyzer.cc behavior for:
  - baseline windowing using config->channels[i].baseline_time fractions
  - optional HNR baseline removal: fit baseline region with const + A*sin(phi0 + omega*t)
    (implemented as linear least-squares with fixed omega = 2*pi/75 ns^-1; same seed omega as C++)
  - channel shift hack for indices i%9==8 and i<=287: shift by 150 samples, fill baseline, myTimeOffset=+30 ns
  - scaling: scale_factor = config.getChannelMultiplicationFactor(i) (polarity included)
  - baseline subtraction and scaling applied sample-by-sample (like C++)
  - peak picking (idx_min/amp) with counter_auto_pol_switch rule (abs vs min)
  - fittable logic thresholds
  - auto polarity switch block behavior when counter_auto_pol_switch>0 (applied per-event; no shared mutation)
  - GetIdxFirstCross walking logic
  - LP2_50 via analytical polynomial solver analogue (time as function of amplitude)
  - optional gaussian fit window selection and myTimeOffset
  - optional linear rising-edge fit and myTimeOffset

Outputs:
  - Minimal (default): per-channel
        <channel>_t_peak
        <channel>_LP2_50

  - If --all_features: additionally per-channel
        <channel>_gaus_mean
        <channel>_gaus_sigma
        <channel>_gaus_chi2
        <channel>_linear_RE_<NN>        for NN=int(100*f) over config.constant_fraction
        <channel>_linear_RE__<TT>mV     for TT=int(abs(thr)) over config.constant_threshold

Dependencies:
  - numpy, uproot, awkward, tqdm
  - scipy optional (curve_fit for gaussian); otherwise moment fallback
"""

import argparse
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import uproot
import awkward as ak
from tqdm import tqdm
import multiprocessing as mp

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
DT_NS = 0.2  # ns per sample; C++: time[i] = (200.0/1000.0)*i
CHANNEL_RE = re.compile(r"^DRS_Board\d+_Group\d+_Channel\d+$")

# C++ shift hack list equals: i%9==8 and i<=287
SHIFT_MOD = 9
SHIFT_REM = 8
SHIFT_MAX_I = 287
SHIFT_SAMPLES = 150

# C++ HNR initial omega seed
HNR_OMEGA = 2.0 * np.pi / 75.0  # rad/ns


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
    constant_fraction: List[float] = field(default_factory=list)
    constant_threshold: List[float] = field(default_factory=list)
    verbose: bool = False

    def has_channel(self, i: int) -> bool:
        return i in self.channels

    def get_mult_factor(self, i: int) -> float:
        """
        Mimic config->getChannelMultiplicationFactor(i).
        Your earlier convention: polarity * 10^(amp/20) * 10^(-att/20)
        """
        c = self.channels[i]
        pol = float(c.polarity)
        gain = 10.0 ** (c.amplification / 20.0)
        att = 10.0 ** (-c.attenuation / 20.0)
        return pol * gain * att


# ---------------------------
# Config parser (robust to multiple naming styles)
# ---------------------------
def _to_fraction_maybe(x: float) -> float:
    # Accept "50" meaning 0.50, or "0.5" meaning 0.5
    return x / 100.0 if x > 1.0 else x


def _parse_baseline_tokens_auto(a: float, b: float) -> Tuple[float, float]:
    """
    Supports:
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
    """
    Tries to follow the TimingDAQ config style used by your C++ code:
      - ConstantFraction / ConstantThreshold lines (case-insensitive)
      - baseline lines
      - channel definition lines:
            CH  POLARITY  AMPLIFICATION  ATTENUATION  ALGORITHM  FILTER_WIDTH

    Also supports underscore variants: constant_fraction, constant_threshold.
    """
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

            key = toks[0].strip()

            # ---- Globals (ConstantFraction / ConstantThreshold) ----
            if key.lower() in ("constantfraction", "constant_fraction"):
                vals: List[float] = []
                for x in toks[1:]:
                    try:
                        v = float(x)
                        vals.append(_to_fraction_maybe(v))
                    except Exception:
                        pass
                if vals:
                    cfg.constant_fraction = vals
                continue

            if key.lower() in ("constantthreshold", "constant_threshold"):
                vals: List[float] = []
                for x in toks[1:]:
                    try:
                        vals.append(float(x))
                    except Exception:
                        pass
                cfg.constant_threshold = vals
                continue

            # ---- Baseline lines ----
            if key.lower() == "baseline" and len(toks) >= 4:
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

            # ---- Per-channel definition ----
            if toks[0].isdigit() and len(toks) >= 6:
                ch = int(toks[0])

                pol_tok = toks[1]
                if pol_tok == "+":
                    polarity = +1
                elif pol_tok == "-":
                    polarity = -1
                else:
                    try:
                        polarity = int(pol_tok)
                    except Exception:
                        polarity = +1

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

                # Re bounds if present: Re##-##
                m = re_rebounds.search(algorithm)
                if m:
                    c.re_bounds = (int(m.group(1)) / 100.0, int(m.group(2)) / 100.0)

                # G fraction if present: G##
                m = re_gfrac.search(algorithm)
                if m:
                    c.gaus_fraction = int(m.group(1)) / 100.0

                # LP degrees
                for deg in (1, 2, 3):
                    if f"LP{deg}" in algorithm:
                        c.PL_deg.append(deg)

                cfg.channels[ch] = c
                continue

            # unknown line ignored

    # If config doesn't define constant_fraction, match your C++ InitLoop expectation:
    # C++ only uses these if config->constant_fraction has entries.
    if not cfg.constant_fraction:
        # default commonly used in your scripts; safe fallback
        cfg.constant_fraction = [0.15, 0.30, 0.45]

    return cfg


# ---------------------------
# Core algorithms matching C++
# ---------------------------
def get_idx_first_cross(value: float, v: np.ndarray, i_st: int, direction: int) -> int:
    """
    Exact logic from C++ GetIdxFirstCross()
    """
    idx_end = (len(v) - 1) if direction > 0 else 0
    rising = (value > v[int(i_st)])
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
    """
    Equivalent to C++ AnalyticalPolinomialSolver used as:
      AnalyticalPolinomialSolver(N, &channel[..], &time[..], deg, coeff);
    i.e. fit time = f(amplitude).
    """
    X = np.vstack([x_amp ** k for k in range(deg + 1)]).T
    coeff, *_ = np.linalg.lstsq(X, y_time, rcond=None)
    return coeff.astype(float)


def poly_eval(x: float, coeff: np.ndarray) -> float:
    out = 0.0
    for k, c in enumerate(coeff):
        out += c * (x ** k)
    return float(out)


# ---------------------------
# HNR baseline fit (C++ sin fit approximation)
# ---------------------------
def hnr_fit_baseline(time_ns: np.ndarray, raw: np.ndarray, bl_st: int, bl_en: int) -> Tuple[float, np.ndarray]:
    """
    C++ does: f_bl = const + A*sin(phi0 + omega*x) with omega free (seed 2*pi/75).
    Here we fix omega=2*pi/75 and fit const + a*sin(omega*t) + b*cos(omega*t) via linear LS.
    Then baseline const = fitted const, and f_eval(t) computed for subtraction.

    Returns:
      baseline_const (raw units),
      f_eval array over full waveform (raw units)
    """
    t = time_ns[bl_st:bl_en].astype(float)
    y = raw[bl_st:bl_en].astype(float)

    s = np.sin(HNR_OMEGA * t)
    c = np.cos(HNR_OMEGA * t)

    A = np.column_stack([np.ones_like(t), s, c])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)  # [const, a, b]

    const0 = float(beta[0])
    a_sin = float(beta[1])
    b_cos = float(beta[2])

    f_eval = const0 + a_sin * np.sin(HNR_OMEGA * time_ns) + b_cos * np.cos(HNR_OMEGA * time_ns)
    return const0, f_eval.astype(float)


# ---------------------------
# Gaussian fit (C++-like window)
# ---------------------------
def _gaus(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_gaussian_peak(t: np.ndarray, y: np.ndarray, j_down: int, j_up: int, amp: float, baseline_rms: float, frac: float) -> Tuple[float, float, float]:
    """
    Mimic C++ gaussian window and seeding logic as closely as practical.

    C++:
      ext_sigma = time[j_up]-time[j_down]
      if (amp*frac < -baseline_RMS) ext_sigma *= 0.25;
      p0 = amp*sqrt(2*pi)*ext_sigma; p1=time[idx_min]; p2=ext_sigma;
    Note ROOT's "gaus" params: [0]=amplitude(height), [1]=mean, [2]=sigma.
    But their seed [0] is "area-like". ROOT often still converges; in python we seed both ways.
    """
    if j_up <= j_down:
        return 0.0, 0.0, 0.0

    lo = max(0, j_down)
    hi = min(len(t) - 1, j_up)
    if hi - lo + 1 < 5:
        return 0.0, 0.0, 0.0

    x = t[lo:hi + 1]
    yy = y[lo:hi + 1]

    mu0 = float(x[np.argmin(yy)])  # for negative pulse, min is peak
    ext_sigma = float(max(x[-1] - x[0], 1e-6))
    if (amp * frac) < (-baseline_rms):
        ext_sigma *= 0.25
    sigma0 = float(max(ext_sigma, 1e-6))

    # Two reasonable A seeds:
    # (1) C++ style (area-like)
    A0_area = float(amp * np.sqrt(2.0 * np.pi) * sigma0)
    # (2) peak height seed
    A0_height = float(amp)

    if _HAVE_SCIPY:
        # try a couple seeds; keep best chi2
        best = None
        for A0 in (A0_height, A0_area):
            try:
                popt, _ = curve_fit(_gaus, x, yy, p0=[A0, mu0, sigma0], maxfev=6000)
                yhat = _gaus(x, *popt)
                chi2 = float(np.sum((yy - yhat) ** 2))
                mu = float(popt[1])
                sig = float(abs(popt[2]))
                cand = (mu, sig, chi2)
                if best is None or cand[2] < best[2]:
                    best = cand
            except Exception:
                continue
        if best is not None:
            return best

    # fallback: weighted moments using -yy
    w = np.clip(-yy, 0.0, None)
    if float(np.sum(w)) <= 0.0:
        return 0.0, 0.0, 0.0
    mu = float(np.sum(w * x) / np.sum(w))
    sig = float(np.sqrt(np.sum(w * (x - mu) ** 2) / np.sum(w)))
    sig = float(max(sig, 1e-6))
    yhat = _gaus(x, A0_height, mu, sig)
    chi2 = float(np.sum((yy - yhat) ** 2))
    return mu, sig, chi2


# ---------------------------
# Per-event feature computation (C++-matched)
# ---------------------------
def compute_features(
    wf_raw_in: np.ndarray,
    time: np.ndarray,
    cfg: Config,
    ch_idx: int,
    all_features: bool,
) -> Dict[str, float]:
    """
    Returns dict with required features (t_peak, LP2_50) and optional extras.
    Waveform input is RAW ADC counts (vector<float>) like C++ reads.

    Important: We do NOT mutate cfg across events (safe for multiprocessing).
    Where C++ mutates polarity/counter in config, we apply equivalent local inversions.
    """
    c = cfg.channels[ch_idx]
    algo = c.algorithm

    out: Dict[str, float] = {}

    # ------- baseline indices (fractions) -------
    bl_st = int(float(c.baseline_time[0]) * NUM_SAMPLES)
    bl_en = int(float(c.baseline_time[1]) * NUM_SAMPLES)
    bl_st = max(0, min(bl_st, NUM_SAMPLES - 1))
    bl_en = max(bl_st + 1, min(bl_en, NUM_SAMPLES))
    bl_len = bl_en - bl_st

    # Use raw baseline mean before any scale (C++)
    wf_raw = wf_raw_in.astype(float, copy=True)
    baseline = float(np.mean(wf_raw[bl_st:bl_en]))

    # scale factor includes polarity (C++: getChannelMultiplicationFactor)
    scale_factor = float(cfg.get_mult_factor(ch_idx))

    # ------- HNR baseline removal option -------
    # C++ fits only if algorithm contains "HNR", and then uses f->Eval(time[j]) for subtraction,
    # and sets baseline = f->GetParameter(0).
    hnr_eval = None
    if "HNR" in algo:
        baseline, hnr_eval = hnr_fit_baseline(time, wf_raw, bl_st, bl_en)

    # ------- shift hack applied BEFORE subtraction (C++) -------
    myTimeOffset = 0.0
    channel_raw = wf_raw  # this is the local "channel" vector in C++ (raw units)
    if (ch_idx % SHIFT_MOD == SHIFT_REM) and (ch_idx <= SHIFT_MAX_I):
        # C++ creates channelNew filled with baseline (raw units)
        channel_new = np.full(NUM_SAMPLES, baseline, dtype=float)
        if SHIFT_SAMPLES < channel_raw.shape[0]:
            copy_len = min(channel_raw.shape[0] - SHIFT_SAMPLES, NUM_SAMPLES)
            channel_new[:copy_len] = channel_raw[SHIFT_SAMPLES:SHIFT_SAMPLES + copy_len]
        channel_raw = channel_new
        myTimeOffset = SHIFT_SAMPLES * DT_NS  # 30 ns

        # If HNR is active, we need corresponding eval on shifted waveform.
        # In C++ they reassign "channel" vector; f->Eval(time[j]) still uses same time grid.
        # Our subtraction uses hnr_eval(time[j]) so it's fine.

    # ------- baseline subtraction + scale sample-by-sample (C++ loop) -------
    # C++ does subtraction inside the min-finding loop, but effect is simply:
    # if HNR: channel = scale*(raw - f_eval(time))
    # else  : channel = scale*(raw - baseline)
    if hnr_eval is not None:
        channel = scale_factor * (channel_raw - hnr_eval)
    else:
        channel = scale_factor * (channel_raw - baseline)

    # baseline_RMS computed on scaled, baseline-subtracted samples in baseline window (C++)
    baseline_RMS = float(np.sqrt(np.mean(channel[bl_st:bl_en] ** 2))) if bl_len > 0 else 0.0

    # ------- peak picking (C++ logic) -------
    idx_min = 0
    amp = 0.0
    for j in range(NUM_SAMPLES):
        range_check = (j > (bl_st + bl_len)) and (j < NUM_SAMPLES)

        if c.counter_auto_pol_switch > 0:
            max_check = abs(channel[j]) > abs(amp)
        else:
            max_check = channel[j] < amp

        if (range_check and max_check) or (j == (bl_st + bl_len)):
            idx_min = j
            amp = float(channel[j])

    out["t_peak"] = float(time[idx_min]) if 0 <= idx_min < NUM_SAMPLES else 0.0

    # ------- fittable (C++ thresholds) -------
    fittable = True
    fittable &= (idx_min < int(NUM_SAMPLES * 0.8))
    fittable &= (abs(amp) > 5.0 * baseline_RMS)
    if 2 <= idx_min <= (NUM_SAMPLES - 3):
        fittable &= (abs(channel[idx_min + 1]) > 2.0 * baseline_RMS)
        fittable &= (abs(channel[idx_min - 1]) > 2.0 * baseline_RMS)
        fittable &= (abs(channel[idx_min + 2]) > 1.0 * baseline_RMS)
        fittable &= (abs(channel[idx_min - 2]) > 1.0 * baseline_RMS)
    else:
        fittable = False

    # ------- auto polarity switch block (C++ behavior as currently written) -------
    # In your C++ snippet, the "if(var[chName+'amp'] < 0)" condition is commented out,
    # so whenever counter_auto_pol_switch>0 and fittable and algorithm not None, they flip.
    # We apply the same *locally* (no global mutation).
    if fittable and ("None" not in algo) and (c.counter_auto_pol_switch > 0):
        amp = -amp
        scale_factor = -scale_factor
        channel = -channel  # flips the waveform in-place effect
        # baseline and baseline storage sign flips in C++; doesn't affect timing results here.

    # ------- LP2_50 (C++ local polynomial fit path for f=0.5, deg=2) -------
    out["LP2_50"] = 0.0
    if fittable and ("None" not in algo) and ("LP2" in algo):
        f = 0.50
        deg = 2

        j_10_pre = get_idx_first_cross(amp * 0.1, channel, idx_min, -1)
        j_90_pre = get_idx_first_cross(amp * 0.9, channel, j_10_pre, +1)

        start_level = -3.0 * baseline_RMS
        j_start = get_idx_first_cross(start_level, channel, idx_min, -1)

        j_st = j_start
        if (amp * f) > start_level:
            j_st = get_idx_first_cross(amp * f, channel, idx_min, -1)

        j_close = get_idx_first_cross(amp * f, channel, j_st, +1)
        if (j_close - 1) >= 0 and abs(channel[j_close - 1] - f * amp) < abs(channel[j_close] - f * amp):
            j_close -= 1

        span_j = int(min(j_90_pre - j_close, j_close - j_st) / 1.5)

        if (j_90_pre - j_10_pre) <= 3 * deg:
            span_j = max(int(deg * 0.5), span_j)
            span_j = max(1, span_j)
        else:
            span_j = max(deg, span_j)

        # C++ also checks bounds and may continue if too short
        if (j_close >= span_j) and (j_close + span_j < NUM_SAMPLES):
            N_add = 1
            if (span_j + N_add + j_close) < j_90_pre:
                N_add += 1

            lo = j_close - span_j
            hi = j_close + span_j + N_add  # exclusive
            if 0 <= lo < hi <= NUM_SAMPLES and (hi - lo) >= (deg + 1):
                coeff = poly_fit_time_as_func_of_amp(channel[lo:hi], time[lo:hi], deg=deg)
                out["LP2_50"] = poly_eval(f * amp, coeff) + myTimeOffset

    if not all_features:
        return out

    # ------- gaussian fit (C++ "G" in algorithm) -------
    out["gaus_mean"] = 0.0
    out["gaus_sigma"] = 0.0
    out["gaus_chi2"] = 0.0
    if fittable and ("G" in algo):
        frac = float(c.gaus_fraction)
        j_down = get_idx_first_cross(amp * frac, channel, idx_min, -1)
        j_up = get_idx_first_cross(amp * frac, channel, idx_min, +1)
        if (j_up - j_down) < 4:
            j_up = min(NUM_SAMPLES - 2, idx_min + 1)
            j_down = max(1, idx_min - 1)

        mu, sig, chi2 = fit_gaussian_peak(time, channel, j_down, j_up, amp, baseline_RMS, frac)
        out["gaus_mean"] = float(mu + myTimeOffset)
        out["gaus_sigma"] = float(sig)
        out["gaus_chi2"] = float(chi2)

    # ------- linear rising edge (C++ "Re" in algorithm) -------
    # C++ uses: i_min = cross(re_bounds[0]*amp, idx_min, -1)
    #           i_max = cross(re_bounds[1]*amp, i_min, +1)
    # then fits y = slope*x + b in [t_min, t_max] on pulse graph
    # and solves t for constant fractions/thresholds.
    if fittable and ("Re" in algo):
        rb0, rb1 = c.re_bounds
        i_min = get_idx_first_cross(rb0 * amp, channel, idx_min, -1)
        i_max = get_idx_first_cross(rb1 * amp, channel, i_min, +1)

        if 0 <= i_min < i_max < NUM_SAMPLES and (i_max - i_min + 1) >= 2:
            x = time[i_min:i_max + 1].astype(float)
            y = channel[i_min:i_max + 1].astype(float)
            # simple linear LS
            m, b = np.polyfit(x, y, 1)
            if m != 0.0 and np.isfinite(m) and np.isfinite(b):
                for ffrac in cfg.constant_fraction:
                    nn = int(round(100.0 * ffrac))
                    out[f"linear_RE_{nn}"] = float((ffrac * amp - b) / m + myTimeOffset)
                for thr in cfg.constant_threshold:
                    tt = int(round(abs(thr)))
                    out[f"linear_RE__{tt}mV"] = float((thr - b) / m + myTimeOffset)
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


# ============================================================
# Multiprocessing chunk worker (parallel over entry ranges)
# ============================================================
_G = {}


def _worker_init(cfg, time_arr, derived_suffixes, all_features,
                 channel_branches, non_channel, read_branches,
                 tree_name, copy_nonchannel, copy_waveforms):
    _G["cfg"] = cfg
    _G["time"] = time_arr
    _G["derived_suffixes"] = derived_suffixes
    _G["all_features"] = all_features
    _G["channel_branches"] = channel_branches
    _G["non_channel"] = non_channel
    _G["read_branches"] = read_branches
    _G["tree_name"] = tree_name
    _G["copy_nonchannel"] = copy_nonchannel
    _G["copy_waveforms"] = copy_waveforms


def _process_entry_range(job):
    """
    job = (input_file, entry_start, entry_stop, tmpdir)

    Returns: (entry_start, entry_stop, tmppath, n_events)
    """
    input_file, entry_start, entry_stop, tmpdir = job

    cfg = _G["cfg"]
    time_arr = _G["time"]
    derived_suffixes = _G["derived_suffixes"]
    all_features = _G["all_features"]
    channel_branches = _G["channel_branches"]
    non_channel = _G["non_channel"]
    read_branches = _G["read_branches"]
    tree_name = _G["tree_name"]
    copy_nonchannel = _G["copy_nonchannel"]
    copy_waveforms = _G["copy_waveforms"]

    tmppath = os.path.join(
        tmpdir,
        f"tmp_{entry_start}_{entry_stop}_{uuid.uuid4().hex}.root"
    )

    with uproot.open(input_file) as f:
        tree = f[tree_name]
        arrays = tree.arrays(
            read_branches,
            entry_start=entry_start,
            entry_stop=entry_stop,
            library="ak",
        )

    n_evt = len(arrays[read_branches[0]])
    out_chunk = {}

    # Optionally copy non-channel
    if copy_nonchannel:
        for k in non_channel:
            if k in arrays:
                out_chunk[k] = np.asarray(arrays[k])

    # Optionally copy waveforms (big)
    if copy_waveforms:
        for chname in channel_branches:
            out_chunk[chname] = ak.to_list(arrays[chname])

    # Derived branches per channel
    for i, chname in enumerate(channel_branches):
        # allocate
        chunk_out = {suf: np.zeros(n_evt, dtype=np.float32) for suf in derived_suffixes}

        # config skip => leave zeros
        if not cfg.has_channel(i):
            for suf in derived_suffixes:
                out_chunk[ch_out(chname, suf)] = chunk_out[suf]
            continue

        wf_jag = arrays[chname]
        wf_fixed = ak.pad_none(wf_jag[:, :NUM_SAMPLES], NUM_SAMPLES, clip=True)
        wf_np = ak.to_numpy(wf_fixed)
        wf_np = np.asarray(wf_np)
        if np.ma.isMaskedArray(wf_np):
            wf_np = wf_np.filled(np.nan)
        wf_np = wf_np.astype(np.float32, copy=False)

        # loop events
        for e in range(n_evt):
            wf = wf_np[e]
            if not np.isfinite(wf).any():
                continue
            finite = np.isfinite(wf)
            if not finite.all():
                last = float(wf[finite][-1]) if finite.any() else 0.0
                wf = np.where(finite, wf, last).astype(np.float32, copy=False)

            feats = compute_features(
                wf_raw_in=wf,
                time=time_arr,
                cfg=cfg,
                ch_idx=i,
                all_features=all_features,
            )
            for suf in derived_suffixes:
                chunk_out[suf][e] = float(feats.get(suf, 0.0))

        for suf in derived_suffixes:
            out_chunk[ch_out(chname, suf)] = chunk_out[suf]

    # time branch (store once per event as in the C++ output tree)
    out_chunk["time"] = np.tile(time_arr, (n_evt, 1))

    # write temp
    with uproot.recreate(tmppath) as fout:
        fout[tree_name] = out_chunk

    return (entry_start, entry_stop, tmppath, n_evt)


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="TimingDAQ-like ntuplizer (multiprocessing over entry ranges), C++-matched logic."
    )
    ap.add_argument("-i", "--input", required=True, help="Input ROOT file")
    ap.add_argument(
        "-o", "--output", required=True,
        help="Output ROOT filename ONLY (no path). Use --outdir to control directory."
    )
    ap.add_argument("--outdir", default=".", help="Output directory (default: .)")
    ap.add_argument("--tag", default="", help="Tag appended to output filename (before .root)")
    ap.add_argument("--tree", default="EventTree", help="Tree name (default: EventTree)")
    ap.add_argument(
        "--config",
        default="/lustre/research/hep/cmadrid/TimingDAQ/DRS_Service_TTU_2025_Sep.config",
        help="TimingDAQ config file path.",
    )
    ap.add_argument("--all_features", action="store_true",
                    help="Also compute gaus_* and linear_RE_* branches.")
    ap.add_argument("--chunk", type=int, default=2000,
                    help="Events per chunk (default: 2000)")
    ap.add_argument("--nproc", type=int, default=0,
                    help="Processes (default: 0 -> cpu_count)")
    ap.add_argument("--tmpdir", default="",
                    help="Temp directory (default: <outdir>/.tmp_ntuple)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--copy_nonchannel", action="store_true",
                    help="Copy non-channel branches to output")
    ap.add_argument("--copy_waveforms", action="store_true",
                    help="Copy waveform vector branches to output (large)")
    ap.add_argument("--keep_tmp", action="store_true",
                    help="Keep temp chunk ROOT files (debug)")
    args = ap.parse_args()

    # ----- output path (filename only + tag) -----
    outname = args.output
    base, ext = os.path.splitext(outname)
    if not ext:
        ext = ".root"
    if args.tag:
        outname = f"{base}_{args.tag}{ext}"
    else:
        outname = f"{base}{ext}"

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, outname)

    tmpdir = args.tmpdir.strip() or os.path.join(args.outdir, ".tmp_ntuple")
    os.makedirs(tmpdir, exist_ok=True)

    # ----- config + time -----
    cfg = parse_config_file(args.config, verbose=args.verbose)
    time_arr = (DT_NS * np.arange(NUM_SAMPLES)).astype(np.float32)

    # ----- inspect input tree -----
    with uproot.open(args.input) as f:
        tree = f[args.tree]
        keys = list(tree.keys())
        nentries = int(tree.num_entries)

    channel_branches = sorted([k for k in keys if is_channel_branch(k)])
    if not channel_branches:
        raise RuntimeError("No DRS_Board*_Group*_Channel* waveform branches found in tree.")

    non_channel = [k for k in keys if (k not in channel_branches and k != "time")]

    read_branches = list(channel_branches)
    if args.copy_nonchannel:
        read_branches += non_channel

    # ----- derived suffixes -----
    derived_suffixes = ["t_peak", "LP2_50"]
    if args.all_features:
        derived_suffixes += ["gaus_mean", "gaus_sigma", "gaus_chi2"]
        for ffrac in cfg.constant_fraction:
            nn = int(round(100.0 * ffrac))
            derived_suffixes.append(f"linear_RE_{nn}")
        for thr in cfg.constant_threshold:
            tt = int(round(abs(thr)))
            derived_suffixes.append(f"linear_RE__{tt}mV")

    # ----- build chunk jobs -----
    if args.chunk <= 0:
        raise ValueError("--chunk must be > 0")

    ranges = [(s, min(s + args.chunk, nentries)) for s in range(0, nentries, args.chunk)]
    jobs = [(args.input, s, e, tmpdir) for (s, e) in ranges]

    nproc = args.nproc if args.nproc and args.nproc > 0 else (mp.cpu_count() or 1)

    if args.verbose:
        print("=" * 90)
        print(" TIMING NTUPLIZER (MULTIPROCESSING, C++-MATCHED)")
        print("=" * 90)
        print(f"Input     : {args.input}")
        print(f"Tree      : {args.tree}")
        print(f"Entries   : {nentries:,}")
        print(f"Channels  : {len(channel_branches)}")
        print(f"Chunk     : {args.chunk} -> {len(jobs)} chunks")
        print(f"nproc     : {nproc}")
        print(f"Features  : {'all' if args.all_features else 'minimal'}")
        print(f"Outpath   : {outpath}")
        print(f"Tmpdir    : {tmpdir}")
        if args.all_features and (not _HAVE_SCIPY):
            print("[INFO] scipy not found; gaussian fit uses moment-based fallback.")
        print("=" * 90)

    # ----- multiprocessing compute -----
    ctx = mp.get_context("fork")  # best for Linux HPC
    results = []
    with ctx.Pool(
        processes=nproc,
        initializer=_worker_init,
        initargs=(
            cfg, time_arr, derived_suffixes, args.all_features,
            channel_branches, non_channel, read_branches,
            args.tree, args.copy_nonchannel, args.copy_waveforms
        ),
    ) as pool:
        for res in tqdm(pool.imap_unordered(_process_entry_range, jobs),
                        total=len(jobs), desc="Chunks (compute)"):
            results.append(res)

    results.sort(key=lambda x: x[0])

    # ----- merge temp files into final output -----
    tree_written = False
    with uproot.recreate(outpath) as fout:
        for (_, _, tmppath, _) in tqdm(results, desc="Chunks (merge)"):
            with uproot.open(tmppath) as tf:
                ttmp = tf[args.tree]
                arr = ttmp.arrays(library="ak")

            if not tree_written:
                fout[args.tree] = arr
                tree_written = True
            else:
                fout[args.tree].extend(arr)

    # ----- cleanup -----
    if not args.keep_tmp:
        for (_, _, tmppath, _) in results:
            try:
                os.remove(tmppath)
            except Exception:
                pass
        try:
            if not os.listdir(tmpdir):
                os.rmdir(tmpdir)
        except Exception:
            pass

    print(f"\nWrote: {outpath}")
    print(f"Tree : {args.tree}")
    print(f"--all_features: {args.all_features}")
    print(f"Chunks: {len(results)}  nproc: {nproc}")


if __name__ == "__main__":
    main()
