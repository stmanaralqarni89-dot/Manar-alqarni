[README.md](https://github.com/user-attachments/files/27551965/README.md)
# Manar-alqarni# qnn_analysis

**Numerical Stability and Error Propagation Analysis in Quantized Neural Network Inference**

COM-701 Doctoral Research — Milestone 3
[manar alqarni] | [447540251@std.psau.sa]
Department of Computer Science, Prince Sattam bin Abdulaziz University

---

## Overview

`qnn_analysis` is a Python package that implements the complete experimental
framework for the Milestone 3 research report. It provides tools to:

- Quantize neural network weights to INT4, INT8, and FP16 precision
- Profile per-layer numerical sensitivity using Algorithm 1 from the report
- Solve memory-constrained mixed-precision bit-width assignment via a dynamic program
- Evaluate the Theorem 1 deterministic error bound against observed output error
- Reproduce all 6 figures and 4 tables in the report from scratch

All experiments run in approximately 3 minutes on a standard CPU with no GPU required.

---

## Requirements

| Package | Minimum Version |
|---|---|
| Python | 3.9 |
| torch | 2.0 |
| numpy | 1.20 |
| matplotlib | 3.5 |

Optional — for extended experiments with real BERT / GPT-2:

| Package | Minimum Version |
|---|---|
| transformers | 4.30 |

---

## Installation

```bash
# 1. Clone or unpack the source
cd Code/

# 2. Install dependencies
pip install torch numpy matplotlib

# 3. Install the package in editable mode
pip install -e .
```

---

## Project Structure

```
Code/
├── qnn_analysis/
│   ├── core/
│   │   ├── quantizer.py            # INT4 / INT8 / FP16 quantizers + UniformQuantizer
│   │   ├── error_tracker.py        # Forward-hook activation error tracker
│   │   └── sensitivity.py          # Algorithm 1 — layer-wise sensitivity profiler
│   ├── models/
│   │   ├── mlp_bench.py            # 3-layer MLP  784 -> 256 -> 128 -> 10
│   │   └── bert_wrapper.py         # Self-contained BERT-style transformer block
│   ├── experiments/
│   │   ├── sweep_bits.py           # Bit-width sweep driver
│   │   ├── calibrate.py            # PTQ calibration convergence experiment
│   │   ├── mixed_precision_dp.py   # DP solver for mixed-precision assignment
│   │   └── theoretical_bound.py    # Theorem 1 bound calculator
│   ├── utils/
│   │   ├── stats.py                # Accuracy, relative error, Lyapunov index
│   │   └── viz.py                  # Matplotlib figure generators
│   └── tests/
│       ├── test_quantizer.py
│       ├── test_sensitivity.py
│       ├── test_mixed_precision.py
│       └── test_theoretical_bound.py
├── examples/
│   ├── run_mlp_experiment.py
│   ├── run_bert_experiment.py
│   ├── run_calibration_experiment.py
│   ├── run_mixed_precision_experiment.py
│   └── run_all.sh
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Running the Tests

Run each test module individually:

```bash
python -m qnn_analysis.tests.test_quantizer
python -m qnn_analysis.tests.test_sensitivity
python -m qnn_analysis.tests.test_mixed_precision
python -m qnn_analysis.tests.test_theoretical_bound
```

Expected output for each: `[OK] all <name> tests passed`

Or run all tests at once with pytest:

```bash
pip install pytest
pytest qnn_analysis/tests -v
```

All 16 tests should pass.

---

## Reproducing the Report Figures

Each example script writes its figures (PDF) and a `results.json` to
`--output-dir`. The defaults below match the LaTeX report layout.

### Figure 1 — MLP bit-width sweep (Section 5.1, Table 1)

```bash
python examples/run_mlp_experiment.py --output-dir ../Report/figures/mlp
```

Produces `fig_bit_sweep.pdf`, `fig_layer_sensitivity.pdf`,
`fig_bound_vs_observed.pdf`, and `results.json`.
Confirms the O(2^-b) error scaling: FP16=2.11e-4, INT8=9.98e-3, INT4=1.85e-1.

### Figure 2 — Transformer-block sensitivity (Section 5.2, Table 3)

```bash
python examples/run_bert_experiment.py --output-dir ../Report/figures/bert
```

Profiles all 6 sub-layers (W_Q, W_K, W_V, W_O, FFN1, FFN2).
Key finding: W_V has condition number kappa=6,389 vs FFN kappa=2.95 —
a 2,167x disparity.

### Figure 3 — PTQ calibration convergence (Section 5.3)

```bash
python examples/run_calibration_experiment.py \
    --output-dir ../Report/figures/calibration
```

Produces log-log plots of relative quantization error vs. calibration set
size N in {50, ..., 50000} with an O(1/sqrt(N)) reference line.
INT8 follows the reference; INT4 plateaus immediately.

### Figure 4 — Mixed-precision Pareto curve (Section 5.4, Table 4)

```bash
python examples/run_mixed_precision_experiment.py \
    --output-dir ../Report/figures/mixed_precision
```

Sweeps memory budgets 30%-100% of the all-INT16 baseline.
At 40% budget: INT16 attention + INT4 FFN gives 2x compression
at only +15% predicted error.

### Figure 6 - Extended evaluation on standard public benchmarks (Section 4.4)

Run `python qnn_analysis_figures.py` to produce `figures/fig_6_public_benchmarks.png`.

Values are cited from published literature and explicitly marked as such in the figure.
Full end-to-end validation is deferred to future work.

| Benchmark | FP32 Accuracy | INT8 Drop | Source |
|---|---|---|---|
| GLUE/SST-2 (BERT-base) | 93.5% | < 0.50 pp | Dettmers et al. 2022; Xiao et al. 2023 |
| CIFAR-10 (ResNet-18) | ~93.0% | < 0.30 pp | Nagel et al. 2021; Frantar et al. 2023 |

Theorem 1 bounds apply directly to both: BERT-base shares the B2 block architecture
(d_model=768, 12 heads); ResNet residual blocks reduce to sequential matrix
multiplications covered by Theorem 1.

### Run everything at once

```bash
bash examples/run_all.sh
```

---

## Module Reference

### `qnn_analysis.core.quantizer`

Implements uniform symmetric quantization:

```
Q(v) = round(v / s) * s,   s = max|W| / (2^(b-1) - 1)
```

Key names:

| Name | Description |
|---|---|
| `quantize_uniform(tensor, num_bits)` | Apply b-bit quantization, return (dequantized tensor, scale) |
| `quantize_int4(tensor)` | INT4 shortcut |
| `quantize_int8(tensor)` | INT8 shortcut |
| `quantize_fp16(tensor)` | FP16 round-trip cast |
| `UniformQuantizer(num_bits)` | Stateful — calibrate once, reuse across passes |
| `unit_roundoff(precision)` | Returns u for `"fp32"`, `"fp16"`, `"int8"`, `"int4"` |

```python
from qnn_analysis.core.quantizer import UniformQuantizer

W = torch.randn(256, 256)
q = UniformQuantizer(num_bits=8)
q.calibrate(W)
W_q = q.quantize(W)
print(q.quantization_error(W))   # relative Frobenius error
```

---

### `qnn_analysis.core.sensitivity`

Implements Algorithm 1 from the report — the layer-wise sensitivity profiler.

```
s^(l) = e4^(l) * kappa^(l)

where:
  e_b^(l) = ||a_fp32^(l) - a_b^(l)||_F  /  ||a_fp32^(l)||_F
  kappa^(l) = sigma_max(W^(l)) / sigma_min(W^(l))
```

```python
from qnn_analysis.core.sensitivity import SensitivityProfiler

profiler = SensitivityProfiler(model, layer_names, bit_widths=(4, 8, 16))
result = profiler.run(x)
print(profiler.to_table())
```

---

### `qnn_analysis.experiments.mixed_precision_dp`

Solves the memory-constrained mixed-precision assignment problem:

```
min   sum_l  e_{b_l}^(l) * kappa^(l)
s.t.  sum_l  p_l * b_l  <=  M
      b_l in B = {4, 8, 16}
```

via a 1-D knapsack dynamic program in O(L x |B| x M/delta) time.

```python
from qnn_analysis.experiments.mixed_precision_dp import solve_mixed_precision

result = solve_mixed_precision(
    layer_names       = ["fc1", "fc2", "fc3"],
    param_counts      = [200960, 32896, 1290],
    sensitivities     = {...},   # from SensitivityProfiler
    condition_numbers = {...},
    memory_budget     = 5e7,     # bits
    bit_options       = (4, 8, 16),
    granularity       = 4096.0,
)
print(result.bit_widths)    # e.g. {"fc1": 8, "fc2": 8, "fc3": 16}
print(result.total_memory)  # bits
print(result.total_error)   # predicted error
```

---

### `qnn_analysis.experiments.theoretical_bound`

Evaluates the Theorem 1 deterministic bound:

```
||f(x) - f_hat(x)|| <= sum_l  eps^(l)  *  prod_{j > l}  K^(j)

where:
  eps^(l) = ||W^(l) - Q(W^(l))||_F    (quantization error norm)
  K^(j)   = ||W^(j)||_2               (spectral norm, i.e. sigma_max)
```

```python
from qnn_analysis.experiments.theoretical_bound import compute_layered_bound

bound = compute_layered_bound(model, layer_names, x, num_bits=8)
print(bound.per_layer_bounds)  # each layer's contribution
print(bound.total_bound)       # sum over all layers
print(bound.tightness_ratio)   # bound / observed
```

---

## Key Results Summary

| Experiment | Key Finding |
|---|---|
| MLP sweep (B1) | O(2^-b) scaling confirmed: FP16=2.11e-4, INT8=9.98e-3, INT4=1.85e-1 |
| Transformer sensitivity (B2) | W_V kappa=6389 vs FFN kappa=2.95 — a **2,167x disparity** |
| PTQ calibration INT8 (B3) | Follows O(1/sqrt(N)), saturates at N=10,000 |
| PTQ calibration INT4 (B3) | Plateaus immediately — more data does not help at 4 bits |
| DP solver at 40% budget (B4) | **2x memory compression** at +15% predicted error |
| Theorem 1 bound | Valid with tightness ratio rho=182 (deterministic); rho=8 (probabilistic) |

---

## Reproducibility Notes

- All randomness is controlled by explicit `seed` parameters (default: 42).
- `torch.manual_seed(seed)` and `numpy.random.seed(seed)` are called at the start of each experiment.
- SVD computations fall back to CPU when GPU SVD is non-deterministic.
- Each experiment writes a `results.json` file with all numerical values used in the report tables and figures.


