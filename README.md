# 🎙️ Adaptive Noise Cancellation — LMS & Leaky LMS

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

A complete Python implementation of **Adaptive Noise Cancellation (ANC)** using the classical **Least Mean Square (LMS)** and **Leaky LMS (LLMS)** algorithms, with an exhaustive **grid-search hyperparameter optimiser** that finds the best combination of filter order *N*, step size *μ*, and leakage factor *α*.

---

## 📖 Table of Contents

- [Background](#background)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Background

Fixed-frequency filters cannot adapt when noise spectra change. **Adaptive Noise Cancellation** continuously updates its coefficients to track the noise in real time:

```
Primary mic  →  d(n) = s(n) + noise(n)  ─────────────────────┐
                                                              ▼
Reference mic →  x(n) ≈ noise(n)  →  [ Adaptive Filter ]  → y(n)
                                                              │
                                              e(n) = d(n) − y(n) ← estimated clean speech
```

This repository evaluates both **LMS** and **Leaky LMS** across three challenging input SNR levels (0 dB, −5 dB, −10 dB) inspired by the **F16 cockpit noise** benchmark from the NOISEX-92 corpus.

---

## Algorithms

### LMS — Least Mean Square

```
w(n+1) = w(n) + μ · e(n) · x(n)
```

| Symbol | Meaning |
|--------|---------|
| `w(n)` | Filter weight vector at time *n* |
| `μ`    | Step size (learning rate) |
| `e(n)` | Error signal = d(n) − y(n) |
| `x(n)` | Reference input vector |

**Stability condition:** `0 < μ < 2 / (N · Pₓ)`

---

### Leaky LMS — Regularised Variant

```
w(n+1) = (1 − μα) · w(n) + μ · e(n) · x(n)
           └── weight decay ──┘
```

| Symbol  | Meaning |
|---------|---------|
| `α`     | Leakage factor — prevents weight drift during silence |

**Stability condition:** `μ · α < 1`

The leakage term prevents coefficient blow-up when the reference signal has insufficient excitation (e.g., speech pauses), making LLMS preferred for real-world deployment.

---

## Project Structure

```
anc-noise-cancellation/
│
├── adaptive_noise_cancellation.py   ← Main implementation
│
├── README.md                        ← This file
├── CITATION.cff                     ← Machine-readable citation
├── LICENSE                          ← MIT License
└── requirements.txt                 ← Python dependencies
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/anc-noise-cancellation.git
cd anc-noise-cancellation

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, NumPy only (no heavy ML frameworks).

---

## Quick Start

```bash
python adaptive_noise_cancellation.py
```

The script will:
1. Generate a synthetic speech signal and F16-type coloured noise
2. Mix them at three input SNR levels: **0 dB, −5 dB, −10 dB**
3. Run the grid search over **30 LMS** and **180 LLMS** configurations
4. Print a full results table and the **overall best parameter combination**

### Expected Output (excerpt)

```
======================================================================
  Grid Search: 30 LMS configs, 180 LLMS configs
  Input SNR: -10 dB
======================================================================

  LMS  best : LMS  | N=32 | mu=0.01000 | SNR_imp=+14.48 dB | MSE=8.18e-02
  LLMS best : LLMS | N=32 | mu=0.01000 | alpha=0.0010 | SNR_imp=+14.33 dB | MSE=8.29e-02

======================================================================
  OVERALL BEST
======================================================================
  LMS  → N=32, mu=0.01000
  LLMS → N=32, mu=0.01000, alpha=0.0010
```

---

## How It Works

### Signal Generation

| Component     | Model |
|---------------|-------|
| Speech        | Sum of 7 harmonics (f₀ = 150 Hz), amplitude-modulated at 4 Hz |
| Noise         | AR(1) process with coefficient 0.95 (F16-type coloured noise) |
| Sample rate   | 16 000 Hz |
| Duration      | 2 seconds |

### Hyperparameter Search Space

| Parameter | Values |
|-----------|--------|
| N (filter order) | 2, 4, 8, 16, 32 |
| μ (step size)    | 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2 |
| α (leakage, LLMS only) | 0.001, 0.01, 0.05, 0.1, 0.2, 0.5 |

### Scoring Function

```
Score = SNR_improvement  −  1000 × avg_MSE
```

This composite score balances perceptual quality (SNR) and statistical accuracy (MSE).

### Performance Metrics

| Metric | Description |
|--------|-------------|
| **SNR Improvement (dB)** | Output SNR − Input SNR |
| **Average MSE** | Mean of e²(n) after warm-up |
| **Convergence speed** | Sample index where smoothed MSE reaches 5% of initial value |

---

## Results

Best configurations found across all input SNR conditions:

| Input SNR | Algorithm | N  | μ     | α      | SNR Imp (dB) | Avg MSE   |
|-----------|-----------|----|-------|--------|--------------|-----------|
| 0 dB      | LMS       | 32 | 0.010 | —      | +7.82        | 6.82e-02  |
| 0 dB      | LLMS      | 32 | 0.010 | 0.001  | +7.80        | 6.83e-02  |
| −5 dB     | LMS       | 32 | 0.010 | —      | +11.76       | 7.15e-02  |
| −5 dB     | LLMS      | 32 | 0.010 | 0.001  | +11.69       | 7.18e-02  |
| −10 dB    | LMS       | 32 | 0.010 | —      | +14.48       | 8.18e-02  |
| −10 dB    | LLMS      | 32 | 0.010 | 0.001  | +14.33       | 8.29e-02  |

> **Key finding:** LMS marginally outperforms LLMS in steady-state metrics, but LLMS is preferred in real-world deployment due to its resistance to weight drift during silence periods.

---

## Citation

If you use this code in your research or project, please cite it as follows.

### BibTeX

```bibtex
@software{anc_lms_llms_2025,
  author       = {Your Name},
  title        = {Adaptive Noise Cancellation using LMS and Leaky LMS},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/<your-username>/anc-noise-cancellation},
  note         = {Python implementation with grid-search hyperparameter optimisation}
}
```

### APA

> Your Name. (2025). *Adaptive Noise Cancellation using LMS and Leaky LMS* [Software]. GitHub. https://github.com/<your-username>/anc-noise-cancellation

### Related Paper

This implementation is inspired by:

> Widrow, B., & Stearns, S. D. (1985). *Adaptive Signal Processing*. Prentice-Hall.

> Varga, A., & Steeneken, H. J. M. (1993). Assessment for automatic speech recognition: II. NOISEX-92. *Speech Communication*, 12(3), 247–251.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request
