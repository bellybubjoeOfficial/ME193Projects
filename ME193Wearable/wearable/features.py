"""Stage 2 feature extraction: gravity separation, windowing, per-window features.

Pipeline for one recording (a DataFrame in the imu.CSV_COLUMNS format):

    accel (g), gyro (°/s)
      -> gravity  = 0.3 Hz low-pass of accel (zero phase, whole recording)
         body     = accel - gravity          (the motion part)
      -> 2.56 s windows, 50 % overlap
      -> one row of features per window       (recording_features)

Gravity is filtered over the WHOLE recording before cutting windows: a 0.3 Hz
filter needs several seconds to settle, longer than one window. (For live
inference in Stage 6, filter a rolling buffer longer than one window.)

Every feature is a plain number with no NaNs: flat signals (std ~ 0) give 0
for skewness, correlations, etc. instead of NaN.

scipy is optional: the App Lab container (where live inference runs) only has
numpy + pandas, so the gravity filter has a numpy version that gives the same
numbers as scipy's (tests/test_stage6_model2.py checks this).
"""

import math

import numpy as np
import pandas as pd

try:
    from scipy import signal
except ImportError:          # App Lab container: use the numpy filter below
    signal = None

from wearable import config, imu

EPS = 1e-9


# ---- Filtering + windowing --------------------------------------------------

def butter2_lowpass(cutoff_hz, fs):
    """2nd-order Butterworth low-pass as one biquad: (b0, b1, b2, a1, a2).

    Standard bilinear-transform formula with frequency pre-warping; it is
    what scipy.signal.butter(2, cutoff_hz, fs=fs) computes.
    """
    k = math.tan(math.pi * cutoff_hz / fs)
    norm = 1 / (1 + math.sqrt(2) * k + k * k)
    b0 = k * k * norm
    return b0, 2 * b0, b0, 2 * (k * k - 1) * norm, (1 - math.sqrt(2) * k + k * k) * norm


def _biquad(x, coef, x0):
    """Run the biquad over list x, starting in steady state for input x0."""
    b0, b1, b2, a1, a2 = coef
    z1, z2 = (1 - b0) * x0, (b2 - a2) * x0      # steady state for a constant input (DC gain = 1)
    y = [0.0] * len(x)
    for i, xi in enumerate(x):
        yi = b0 * xi + z1
        z1 = b1 * xi - a1 * yi + z2
        z2 = b2 * xi - a2 * yi
        y[i] = yi
    return y


def filtfilt_numpy(x, coef, padlen=9):
    """Zero-phase filter of a 1-D signal, the same way scipy.signal.sosfiltfilt does it.

    Pads both ends with an odd reflection (padlen = scipy's default for one
    biquad), filters forward, then backward, then removes the padding.
    """
    x = np.asarray(x, dtype=float)
    padlen = min(padlen, len(x) - 1)
    ext = np.concatenate([2 * x[0] - x[padlen:0:-1], x, 2 * x[-1] - x[-2:-padlen - 2:-1]])
    y = _biquad(ext.tolist(), coef, ext[0])
    y = _biquad(y[::-1], coef, y[-1])[::-1]
    return np.array(y[padlen:len(y) - padlen])


def gravity_lowpass(acc, fs=config.SAMPLE_RATE_HZ, cutoff_hz=config.GRAVITY_CUTOFF_HZ,
                    order=config.GRAVITY_FILTER_ORDER, use_scipy=True):
    """Gravity estimate = zero-phase Butterworth low-pass of accel, shape (n, 3)."""
    acc = np.asarray(acc, dtype=float)
    if use_scipy and signal is not None:
        sos = signal.butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
        return signal.sosfiltfilt(sos, acc, axis=0)
    if order != 2:
        raise ValueError("the numpy filter only does order 2; install scipy for other orders")
    coef = butter2_lowpass(cutoff_hz, fs)
    return np.column_stack([filtfilt_numpy(acc[:, i], coef) for i in range(acc.shape[1])])


def window_starts(n_samples, size=config.WINDOW_SAMPLES, overlap=config.WINDOW_OVERLAP):
    """Start index of every full window. Partial windows at the end are dropped."""
    step = max(1, int(round(size * (1 - overlap))))
    if n_samples < size:
        return np.array([], dtype=int)
    return np.arange(0, n_samples - size + 1, step)


# ---- Single-signal helpers (x = 1-D window) ---------------------------------

def _spectrum(x, fs):
    """Power spectrum of the mean-removed, Hann-windowed signal (zero-padded)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    nfft = max(config.FFT_MIN_POINTS, 1 << (len(x) - 1).bit_length())
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fs)
    return f, p


def dominant_frequency(x, fs=config.SAMPLE_RATE_HZ, f_range=config.DOM_FREQ_RANGE_HZ):
    """Frequency (Hz) of the strongest peak in f_range. 0 for a flat signal."""
    if np.std(x) < EPS:
        return 0.0
    f, p = _spectrum(x, fs)
    band = (f >= f_range[0]) & (f <= f_range[1])
    return float(f[band][np.argmax(p[band])])


def band_energies(x, fs=config.SAMPLE_RATE_HZ, bands=config.SPECTRAL_BANDS_HZ):
    """Signal variance split into frequency bands (units of x²).

    The spectrum is scaled so all bins above 0 Hz add up to var(x); each band
    then gets its share. A band edge belongs to the band below it.
    """
    var = float(np.var(x))
    if var < EPS:
        return [0.0] * len(bands)
    f, p = _spectrum(x, fs)
    p = p[f > 0] * var / np.sum(p[f > 0])
    f = f[f > 0]
    return [float(np.sum(p[(f > lo) & (f <= hi)])) for lo, hi in bands]


def spectral_entropy(x, fs=config.SAMPLE_RATE_HZ):
    """How spread out the spectrum is: ~0 = one pure tone, 1 = flat (white noise)."""
    if np.std(x) < EPS:
        return 0.0
    f, p = _spectrum(x, fs)
    p = p[f > 0]
    p = p / np.sum(p)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(len(f) - 1))


def autocorr_peak(x, fs=config.SAMPLE_RATE_HZ, max_freq_hz=5.0):
    """Highest autocorrelation at a lag between 1/max_freq_hz and half the window.

    Near 1 = strongly periodic (steady gait or pedalling), near 0 = irregular.
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    if np.std(x) < EPS:
        return 0.0
    spec = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:n]
    ac = ac / ac[0] * n / (n - np.arange(n))     # unbiased: each lag averaged over its overlap
    lo, hi = int(fs / max_freq_hz), n // 2
    return float(np.clip(np.max(ac[lo:hi + 1]), -1, 1))


def zero_crossing_rate(x, fs=config.SAMPLE_RATE_HZ):
    """Mean-crossings per second (Hz)."""
    s = np.signbit(np.asarray(x, dtype=float) - np.mean(x))
    return float(np.count_nonzero(s[1:] != s[:-1]) * fs / len(x))


# Skewness / kurtosis / correlation written out with numpy: scipy.stats gives
# the same numbers (its defaults: biased, Fisher "excess" kurtosis) but has
# ~5 ms of overhead per call, which dominated the run time on the Q.
def _safe_skew(x):
    d = x - np.mean(x)
    m2 = np.mean(d ** 2)
    return float(np.mean(d ** 3) / m2 ** 1.5) if m2 > EPS ** 2 else 0.0


def _safe_kurtosis(x):
    d = x - np.mean(x)
    m2 = np.mean(d ** 2)
    return float(np.mean(d ** 4) / m2 ** 2 - 3) if m2 > EPS ** 2 else 0.0   # normal = 0


def _safe_corr(a, b):
    da, db = a - np.mean(a), b - np.mean(b)
    den = np.sqrt(np.sum(da ** 2) * np.sum(db ** 2))
    return float(np.sum(da * db) / den) if den > EPS ** 2 else 0.0


def signal_features(x, prefix, fs=config.SAMPLE_RATE_HZ):
    """The standard block of features for one 1-D signal, as a dict."""
    x = np.asarray(x, dtype=float)
    q75, q25 = np.percentile(x, [75, 25])
    feats = {
        "mean": np.mean(x),
        "std": np.std(x),
        "rms": np.sqrt(np.mean(x ** 2)),
        "min": np.min(x),
        "max": np.max(x),
        "range": np.ptp(x),
        "iqr": q75 - q25,
        "skew": _safe_skew(x),
        "kurt": _safe_kurtosis(x),
        "jerk_rms": np.sqrt(np.mean((np.diff(x) * fs) ** 2)),    # units per second
        "zcr": zero_crossing_rate(x, fs),
        "dom_freq": dominant_frequency(x, fs),
        "spec_entropy": spectral_entropy(x, fs),
        "acf_peak": autocorr_peak(x, fs),
    }
    for (lo, hi), e in zip(config.SPECTRAL_BANDS_HZ, band_energies(x, fs)):
        feats[f"band_{lo}_{hi}hz"] = e
    return {f"{prefix}_{k}": float(v) for k, v in feats.items()}


# ---- One window -------------------------------------------------------------

AXES = "xyz"


def window_features(acc, gyr, grav, fs=config.SAMPLE_RATE_HZ):
    """All features for one window. acc, gyr, grav: arrays of shape (n, 3)."""
    body = acc - grav
    acc_mag = np.linalg.norm(acc, axis=1)       # includes gravity (≈1 g when still)
    body_mag = np.linalg.norm(body, axis=1)
    gyr_mag = np.linalg.norm(gyr, axis=1)

    f = {}
    f.update(signal_features(acc_mag, "acc_mag", fs))
    f.update(signal_features(body_mag, "body_mag", fs))
    f.update(signal_features(gyr_mag, "gyr_mag", fs))

    for i, ax in enumerate(AXES):
        f[f"body_{ax}_std"] = float(np.std(body[:, i]))
        f[f"gyr_{ax}_rms"] = float(np.sqrt(np.mean(gyr[:, i] ** 2)))
        # Peak acceleration along the axis, gravity included (Model 1 uses
        # one of these). Peak of |a| so it doesn't depend on which way the
        # axis points.
        f[f"acc_{ax}_peak"] = float(np.max(np.abs(acc[:, i])))

    # Signal magnitude area: average total body acceleration on all axes.
    f["body_sma"] = float(np.mean(np.sum(np.abs(body), axis=1)))

    for i, j in [(0, 1), (0, 2), (1, 2)]:
        f[f"body_corr_{AXES[i]}{AXES[j]}"] = _safe_corr(body[:, i], body[:, j])
        f[f"gyr_corr_{AXES[i]}{AXES[j]}"] = _safe_corr(gyr[:, i], gyr[:, j])

    # Posture: which way gravity points in sensor coordinates (unit vector).
    # Upright (stand/walk/run), shin tilted (sit) or lying on your back (air-biking).
    g = np.mean(grav, axis=0)
    g = g / max(np.linalg.norm(g), EPS)
    for i, ax in enumerate(AXES):
        f[f"grav_{ax}"] = float(g[i])

    # Cadence: dominant frequency of the gyro axis that moves the most (the
    # leg's swing / pedalling axis). The gyro MAGNITUDE can't give this: |sin|
    # repeats twice per cycle, so its spectrum peaks at double the cadence.
    main = int(np.argmax(np.var(gyr, axis=0)))
    f["gyr_main_dom_freq"] = dominant_frequency(gyr[:, main], fs)
    return f


def feature_names():
    """Feature column names in a fixed order (computed from a dummy window)."""
    n = config.WINDOW_SAMPLES
    z = np.zeros((n, 3))
    return list(window_features(z, z, z).keys())


# ---- Whole recording ----------------------------------------------------------

def recording_features(df, fs=config.SAMPLE_RATE_HZ, size=config.WINDOW_SAMPLES,
                       overlap=config.WINDOW_OVERLAP):
    """Feature table for one recording: one row per window.

    Extra column t_start_s = window start time (s from the recording start),
    handy for plots. Returns an empty table if the recording is too short.
    """
    acc = df[imu.ACCEL_COLS].to_numpy(dtype=float)
    gyr = df[imu.GYRO_COLS].to_numpy(dtype=float)
    grav = gravity_lowpass(acc, fs) if len(acc) >= size else acc    # too short: no windows anyway
    rows = []
    for s in window_starts(len(df), size, overlap):
        w = slice(s, s + size)
        row = {"t_start_s": s / fs}
        row.update(window_features(acc[w], gyr[w], grav[w], fs))
        rows.append(row)
    return pd.DataFrame(rows, columns=["t_start_s"] + feature_names())


def build_dataset(recordings, fs=config.SAMPLE_RATE_HZ):
    """Stack many recordings into one table for training.

    recordings: iterable of (df, activity, session_id). Adds columns
    activity (as recorded, e.g. "airbike"), label (model class, the same name)
    and session (used for grouped cross-validation).
    """
    parts = []
    for df, activity, session in recordings:
        feats = recording_features(df, fs)
        feats.insert(0, "session", session)
        feats.insert(1, "activity", activity)
        feats.insert(2, "label", config.ACTIVITY_TO_CLASS[activity])
        parts.append(feats)
    return pd.concat(parts, ignore_index=True)
