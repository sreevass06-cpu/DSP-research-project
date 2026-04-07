"""
Adaptive Noise Cancellation (ANC) System
Implements LMS and Leaky LMS (LLMS) algorithms as described in:
"Hardware Co-Simulation of Adaptive Noise Cancellation System using LMS and Leaky LMS Algorithms"

Finds the best combination of:
  - N     : filter order (number of taps)
  - mu    : step size (learning rate)
  - alpha : leakage factor (LLMS only)

Performance metrics:
  - SNR Improvement (dB)
  - Mean Square Error (MSE)
  - Convergence speed (iterations to reach 95% of final MSE reduction)
"""

import numpy as np
from itertools import product
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Signal generation helpers
# ─────────────────────────────────────────────

def generate_speech_signal(duration: float = 2.0, fs: int = 16000, seed: int = 42) -> np.ndarray:
    """
    Simulate a speech-like signal using a sum of sinusoids with amplitude modulation.
    (Substitute for a real .wav file.)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs)) / fs

    # Voiced speech: fundamental + harmonics
    f0 = 150  # fundamental frequency (Hz)
    speech = sum(
        (1 / k) * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
        for k in range(1, 8)
    )
    # Amplitude modulation to mimic syllable rhythm (~4 Hz)
    am = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    speech *= am
    speech /= np.max(np.abs(speech))  # normalize to [-1, 1]
    return speech.astype(np.float64)


def generate_noise(n_samples: int, noise_type: str = "f16", seed: int = 7) -> np.ndarray:
    """
    Simulate an F16-like colored noise (low-frequency dominated).
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples)

    if noise_type == "f16":
        # Simple 1st-order AR filter to add low-frequency dominance
        noise = np.zeros(n_samples)
        noise[0] = white[0]
        for i in range(1, n_samples):
            noise[i] = 0.95 * noise[i - 1] + white[i]
    else:
        noise = white

    noise /= np.max(np.abs(noise))
    return noise.astype(np.float64)


def mix_signal(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Mix speech and noise at a given input SNR level.
    Returns the noisy speech signal d(n).
    """
    p_speech = np.mean(speech ** 2)
    p_noise  = np.mean(noise ** 2)
    scale = np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    return speech + scale * noise


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Compute SNR in dB between clean signal and noisy/residual signal."""
    noise_power = np.mean((clean - noisy) ** 2)
    signal_power = np.mean(clean ** 2)
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)


def snr_improvement(speech: np.ndarray,
                    noisy_input: np.ndarray,
                    anc_output: np.ndarray) -> float:
    """SNR improvement = output SNR - input SNR."""
    snr_in  = compute_snr(speech, noisy_input)
    snr_out = compute_snr(speech, anc_output)
    return snr_out - snr_in


# ─────────────────────────────────────────────
# Core ANC algorithms
# ─────────────────────────────────────────────

def lms_anc(d: np.ndarray,
            x: np.ndarray,
            N: int,
            mu: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Least Mean Square (LMS) Adaptive Noise Canceller.

    Parameters
    ----------
    d : desired signal = speech + noise  (shape: [n_samples])
    x : reference noise signal           (shape: [n_samples])
    N : filter order (number of taps)
    mu: step size

    Returns
    -------
    e  : error (estimated clean speech)  (shape: [n_samples])
    mse: instantaneous squared error     (shape: [n_samples])
    """
    n_samples = len(d)
    w = np.zeros(N)          # filter weights
    e = np.zeros(n_samples)
    mse = np.zeros(n_samples)

    for n in range(N - 1, n_samples):
        x_vec = x[n:n - N:-1] if N > 1 else np.array([x[n]])  # [x(n), x(n-1), ..., x(n-N+1)]
        x_vec = x[n: n - N: -1] if N > 1 else x[n:n + 1]

        # Safer slicing
        start = n - N + 1
        x_vec = x[n: start - 1: -1] if start > 0 else np.concatenate([np.zeros(N - n - 1), x[:n + 1]])

        y      = np.dot(w, x_vec)    # filter output
        e[n]   = d[n] - y            # error signal
        mse[n] = e[n] ** 2

        # Weight update: w(n+1) = w(n) + mu * e(n) * x(n)
        w += mu * e[n] * x_vec

    return e, mse


def llms_anc(d: np.ndarray,
             x: np.ndarray,
             N: int,
             mu: float,
             alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Leaky Least Mean Square (LLMS) Adaptive Noise Canceller.

    Weight update: w(n+1) = (1 - mu*alpha)*w(n) + mu*e(n)*x(n)

    Parameters
    ----------
    d     : desired signal = speech + noise
    x     : reference noise signal
    N     : filter order
    mu    : step size
    alpha : leakage factor  (0 < alpha < 1)

    Returns
    -------
    e  : error (estimated clean speech)
    mse: instantaneous squared error
    """
    n_samples = len(d)
    w = np.zeros(N)
    e = np.zeros(n_samples)
    mse = np.zeros(n_samples)

    leak = 1.0 - mu * alpha  # leakage term

    for n in range(n_samples):
        start = n - N + 1
        if start >= 0:
            x_vec = x[n: start - 1: -1] if start > 0 else x[n::-1]
        else:
            pad   = np.zeros(-start)
            x_vec = np.concatenate([x[n::-1], pad])

        y      = np.dot(w, x_vec)
        e[n]   = d[n] - y
        mse[n] = e[n] ** 2

        # Leaky weight update
        w = leak * w + mu * e[n] * x_vec

    return e, mse


def _build_x_vec(x: np.ndarray, n: int, N: int) -> np.ndarray:
    """Helper: extract [x(n), x(n-1), ..., x(n-N+1)] with zero-padding."""
    if n >= N - 1:
        # Enough history: slice from n down to n-N+1 (inclusive)
        return x[n - N + 1: n + 1][::-1]
    else:
        # Not enough history: zero-pad the missing past samples
        available = x[: n + 1][::-1]          # [x(n), x(n-1), ..., x(0)]
        pad = np.zeros(N - len(available))
        return np.concatenate([available, pad])


def lms_anc_fast(d: np.ndarray,
                 x: np.ndarray,
                 N: int,
                 mu: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised-friendly LMS using the helper."""
    n_samples = len(d)
    w   = np.zeros(N)
    e   = np.zeros(n_samples)
    mse = np.zeros(n_samples)

    for n in range(n_samples):
        xv     = _build_x_vec(x, n, N)
        e[n]   = d[n] - np.dot(w, xv)
        mse[n] = e[n] ** 2
        w     += mu * e[n] * xv

    return e, mse


def llms_anc_fast(d: np.ndarray,
                  x: np.ndarray,
                  N: int,
                  mu: float,
                  alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised-friendly LLMS using the helper."""
    n_samples = len(d)
    w    = np.zeros(N)
    e    = np.zeros(n_samples)
    mse  = np.zeros(n_samples)
    leak = 1.0 - mu * alpha

    for n in range(n_samples):
        xv     = _build_x_vec(x, n, N)
        e[n]   = d[n] - np.dot(w, xv)
        mse[n] = e[n] ** 2
        w      = leak * w + mu * e[n] * xv

    return e, mse


def convergence_speed(mse: np.ndarray, window: int = 100) -> int:
    """
    Returns the sample index at which the smoothed MSE first reaches
    5 % of its initial value (proxy for convergence speed).
    Lower index = faster convergence.
    """
    smoothed = np.convolve(mse, np.ones(window) / window, mode="valid")
    initial  = smoothed[0]
    target   = 0.05 * initial
    hits = np.where(smoothed <= target)[0]
    return int(hits[0]) if len(hits) > 0 else len(smoothed)


# ─────────────────────────────────────────────
# Result data class
# ─────────────────────────────────────────────

@dataclass
class ANCResult:
    algorithm:   str
    N:           int
    mu:          float
    alpha:       Optional[float]
    snr_input:   float
    snr_improve: float
    avg_mse:     float
    conv_speed:  int
    score:       float = field(init=False)

    def __post_init__(self):
        # Composite score: higher SNR improvement and lower MSE is better.
        # Normalise so both contribute equally (weights tunable).
        self.score = self.snr_improve - 1000 * self.avg_mse

    def __repr__(self):
        a = f"{self.alpha:.4f}" if self.alpha is not None else "  N/A  "
        return (
            f"{self.algorithm:<6} | N={self.N:2d} | mu={self.mu:.5f} | alpha={a} | "
            f"SNR_imp={self.snr_improve:+.4f} dB | MSE={self.avg_mse:.2e} | "
            f"Conv={self.conv_speed:6d} | Score={self.score:.4f}"
        )


# ─────────────────────────────────────────────
# Grid-search optimiser
# ─────────────────────────────────────────────

def grid_search(speech:      np.ndarray,
                noise_ref:   np.ndarray,
                noisy_speech: np.ndarray,
                snr_input_db: float,
                N_values:    list[int],
                mu_values:   list[float],
                alpha_values: list[float],
                verbose:     bool = True) -> tuple[ANCResult, ANCResult, list, list]:
    """
    Exhaustive grid search over (N, mu) for LMS and (N, mu, alpha) for LLMS.

    Returns best_lms, best_llms, all_lms_results, all_llms_results.
    """
    lms_results  = []
    llms_results = []

    total_lms  = len(N_values) * len(mu_values)
    total_llms = len(N_values) * len(mu_values) * len(alpha_values)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Grid Search: {total_lms} LMS configs, {total_llms} LLMS configs")
        print(f"  Input SNR: {snr_input_db} dB")
        print(f"{'='*70}")

    # ── LMS ──────────────────────────────────────────────────────────────
    for idx, (N, mu) in enumerate(product(N_values, mu_values)):
        e, mse = lms_anc_fast(noisy_speech, noise_ref, N, mu)

        # Stability check: skip if weights exploded
        if not np.isfinite(e).all() or np.max(np.abs(e)) > 1e6:
            continue

        snr_imp = snr_improvement(speech, noisy_speech, e)
        avg_mse = float(np.mean(mse[N:]))   # skip warm-up
        conv    = convergence_speed(mse)

        r = ANCResult("LMS", N, mu, None, snr_input_db, snr_imp, avg_mse, conv)
        lms_results.append(r)

    # ── LLMS ─────────────────────────────────────────────────────────────
    for N, mu, alpha in product(N_values, mu_values, alpha_values):
        # Stability condition: mu * alpha < 1
        if mu * alpha >= 1.0:
            continue

        e, mse = llms_anc_fast(noisy_speech, noise_ref, N, mu, alpha)

        if not np.isfinite(e).all() or np.max(np.abs(e)) > 1e6:
            continue

        snr_imp = snr_improvement(speech, noisy_speech, e)
        avg_mse = float(np.mean(mse[N:]))
        conv    = convergence_speed(mse)

        r = ANCResult("LLMS", N, mu, alpha, snr_input_db, snr_imp, avg_mse, conv)
        llms_results.append(r)

    # Sort by composite score descending
    lms_results.sort(key=lambda r: r.score, reverse=True)
    llms_results.sort(key=lambda r: r.score, reverse=True)

    best_lms  = lms_results[0]  if lms_results  else None
    best_llms = llms_results[0] if llms_results else None

    return best_lms, best_llms, lms_results, llms_results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  Adaptive Noise Cancellation – LMS vs Leaky LMS")
    print("  Parameter Optimisation via Grid Search")
    print("=" * 70)

    # ── Signals ──────────────────────────────────────────────────────────
    FS       = 16000
    DURATION = 2.0   # seconds (keep short for fast grid search)

    speech    = generate_speech_signal(DURATION, FS, seed=42)
    noise_ref = generate_noise(len(speech), noise_type="f16", seed=7)

    # ── Search space ─────────────────────────────────────────────────────
    N_VALUES     = [2, 4, 8, 16, 32]
    MU_VALUES    = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    ALPHA_VALUES = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]

    INPUT_SNRS = [0, -5, -10]   # dB  (matches the paper's test conditions)

    all_best = []

    for snr_db in INPUT_SNRS:
        noisy_speech = mix_signal(speech, noise_ref, snr_db)

        best_lms, best_llms, lms_res, llms_res = grid_search(
            speech, noise_ref, noisy_speech,
            snr_input_db=snr_db,
            N_values=N_VALUES,
            mu_values=MU_VALUES,
            alpha_values=ALPHA_VALUES,
            verbose=True,
        )

        print(f"\n  ── Best results for input SNR = {snr_db} dB ──")

        if best_lms:
            print(f"  LMS  best : {best_lms}")
        if best_llms:
            print(f"  LLMS best : {best_llms}")

        print(f"\n  Top-3 LMS configurations:")
        for r in lms_res[:3]:
            print(f"    {r}")

        print(f"\n  Top-3 LLMS configurations:")
        for r in llms_res[:3]:
            print(f"    {r}")

        all_best.append((snr_db, best_lms, best_llms))

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY – Best Parameter Combinations")
    print("=" * 70)
    header = (f"{'Input SNR':>10} | {'Algo':>5} | {'N':>3} | {'mu':>8} | "
              f"{'alpha':>8} | {'SNR Imp (dB)':>13} | {'Avg MSE':>10}")
    print(header)
    print("-" * len(header))

    for snr_db, b_lms, b_llms in all_best:
        for b in [b_lms, b_llms]:
            if b is None:
                continue
            a_str = f"{b.alpha:.4f}" if b.alpha is not None else "   N/A"
            print(
                f"{snr_db:>10} | {b.algorithm:>5} | {b.N:>3} | {b.mu:>8.5f} | "
                f"{a_str:>8} | {b.snr_improve:>+13.4f} | {b.avg_mse:>10.2e}"
            )

    # ── Overall best across all SNR conditions ───────────────────────────
    print("\n" + "=" * 70)
    print("  OVERALL BEST (averaged score across all input SNR levels)")
    print("=" * 70)

    # Re-run a final evaluation at each SNR for the overall best combo
    from collections import defaultdict
    lms_scores  = defaultdict(list)
    llms_scores = defaultdict(list)

    for snr_db, b_lms, b_llms in all_best:
        if b_lms:
            lms_scores[(b_lms.N, b_lms.mu)].append(b_lms.score)
        if b_llms:
            llms_scores[(b_llms.N, b_llms.mu, b_llms.alpha)].append(b_llms.score)

    if lms_scores:
        best_lms_key  = max(lms_scores, key=lambda k: np.mean(lms_scores[k]))
        print(f"  LMS  overall best  → N={best_lms_key[0]}, mu={best_lms_key[1]:.5f}")

    if llms_scores:
        best_llms_key = max(llms_scores, key=lambda k: np.mean(llms_scores[k]))
        print(f"  LLMS overall best  → N={best_llms_key[0]}, mu={best_llms_key[1]:.5f}, "
              f"alpha={best_llms_key[2]:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
