# Forward-observable component feedback (FOCF)

Code and numerical data for **Overcoming the routing bottleneck in direct-feedback training of vision transformers**.

FOCF builds input-dependent teaching signals from forward-observable attention and feed-forward components. The repository contains the implementations and experiments reported in the paper and Supporting Information: FOCF, BP, DFA, GrAPE, the adapted EBD baseline, the identity-carrier and random-dictionary controls, and the frozen-model routing diagnostics.

## Setup

Use Python 3.9 or 3.10. The original training environment used PyTorch 2.0.0, CUDA 11.7, NumPy 1.23.5 and Pillow 9.4.0 on NVIDIA V100 GPUs.

```bash
git clone https://github.com/McGuinnessChen/focf.git
cd focf
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the reported tables and plots

These commands use the released measurements and require no training or dataset download:

```bash
python scripts/summarize_results.py
python scripts/plot_figure_2.py
python scripts/plot_figure_s1.py
```

Results are written to `outputs/`. The CSV tables contain unrounded means and sample standard deviations over seeds 0, 1 and 2. Figure 2 uses the original numerical arrays; the Figure S1 script plots the original measurements. Figures 1 and 3 are conceptual diagrams, and Table S2 describes baseline implementations rather than numerical experiments.

| Paper item | Released data |
| --- | --- |
| Table 1: classification | `data/classification/per_seed.csv` |
| Table 2: structured-feedback controls | CIFAR-10 FOCF, random-dictionary and identity rows in the same CSV |
| Table 3: output-code budget | `data/output_budget/` |
| Figure 2A: fixed-error credit maps | `data/credit_maps/` |
| Figure 2B–C and Table S1: GrAPE routing | `data/routing/` |
| Figure S1: correction exchange | `data/route_swap/` |

The 69 classification runs each include their final summary and all 100 epoch records. The data format and diagnostic protocols are described in [data/README.md](data/README.md).

## Datasets

Obtain the Python versions of [CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) and the official [Tiny ImageNet archive](https://cs231n.stanford.edu/tiny-imagenet-200.zip). Place them under the repository as follows:

```text
cifar10_data/cifar-10-batches-py/
cifar100_data/cifar-100-python/
tiny_imagenet_data/tiny-imagenet-200.zip
tiny_imagenet_data/tiny-imagenet-200/
```

Keep Tiny ImageNet's original `train/`, `val/images/`, `val/val_annotations.txt`, `wnids.txt` and `words.txt` structure. The loaders use the official validation images as the final Tiny ImageNet evaluation set. Training verifies file sizes and SHA-256 values against `configs/datasets/local_datasets.json`.

| Experiment | Training images | Internal validation | Final evaluation |
| --- | ---: | ---: | ---: |
| CIFAR-10 classification and controls | 50,000 | 0 | 10,000 official test images |
| CIFAR-100 classification | 45,000 | 5,000 | 10,000 official test images |
| Tiny ImageNet classification | 90,000 | 10,000 | 10,000 official validation images |
| CIFAR-10 frozen-model diagnostics | 5,000 | 1,000 | Internal validation only |

The internal split uses a fixed permutation with seed 2027. Model seeds are 0, 1 and 2, paired with feedback seeds 100, 101 and 102. Final classification results use epoch 100.

## Train

Each file in `configs/training/` specifies one reported classification run. For example:

```bash
python train.py --config configs/training/cifar10-focf-seed0.json
python train.py --config configs/training/cifar100-focf-seed0.json
python train.py --config configs/training/tiny_imagenet-focf-seed0.json
```

Use the other configuration files to run BP, DFA, GrAPE, EBD and the two CIFAR-10 controls. `--dry-run` prints the resolved arguments. A short CPU check is available with:

```bash
python train.py --config configs/training/cifar10-focf-seed0.json --device cpu --smoke
```

Training writes configuration, metrics and checkpoints to `runs/`. Completed runs are linked at `checkpoints/<configuration-id>.pt`. Use `--resume path/to/last_checkpoint.pt` to resume a run from this release.

FOCF's cached suffix implementation preserves the local update rule and probe schedule. The original CIFAR-100, Tiny ImageNet and random-dictionary runs switched to this implementation during training; the released entrypoint uses it from the first epoch. Floating-point trajectories may therefore differ from the archived measurements.

## Rerun frozen-model diagnostics

The exact diagnostic measurements are included. To generate new measurements, first train the corresponding files in `configs/diagnostic_training/`. These are the 5,000-image models used for Figure 2, Figure S1 and Table S1; they have a different training split from Table 1.

```bash
for config in configs/diagnostic_training/*.json; do
    python train.py --config "$config"
done
python scripts/analyze_grape_routing_problem.py --config configs/diagnostics/routing.json
python scripts/extract_method_credit_fixed_error.py --config configs/diagnostics/credit_maps.json
python scripts/audit_frozen_route_swap.py --config configs/diagnostics/route_swap-seed0.json
```

For Table 3, train the CIFAR-100 or Tiny ImageNet FOCF configuration and run, for example:

```bash
python scripts/audit_frozen_output_budget.py --config configs/diagnostics/output_budget-cifar100-seed0.json
```

Corresponding seed-1 and seed-2 configurations are provided. Frozen response diagnostics use CUDA. Run timing measurements on an otherwise idle GPU. New diagnostic outputs go to `outputs/diagnostics/`; the archived paper data remain in `data/`.

## Implementation and checks

`src/methods/component_postln/` contains FOCF's component dictionary, forward channel maps and analytic local gradients. `component_scaling/` and `component_scaling_fast/` provide chunked and cached evaluation. The other retained modules implement the reported baselines and shared ViT routines.

```bash
python -m pytest -q
```

The tests check component adjoints, local gradients against autograd, the absence of cross-block reverse differentiation in FOCF, calibration fidelity, EBD objectives, checkpoint restoration, and cached-versus-uncached updates.

Code is distributed under the MIT license. The EBD reference kernel in `tests/reference/ebd_gradients.py` retains the upstream copyright in `LICENSE`. Third-party datasets and the CIFAR example images embedded in the figure arrays retain their original terms. Raw dataset archives and trained checkpoints are not included; numerical results and the code to generate them are provided here.
