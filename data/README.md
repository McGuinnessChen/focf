# Paper data

All values come from the experiments reported in the manuscript or Supporting Information. Accuracies in JSON are fractions; columns ending in `_percent` in CSV are percentages. Standard deviations use the sample definition (`ddof=1`).

- `classification/per_seed.csv`: final epoch-100 results for 21 method–dataset combinations in Table 1 and the two additional CIFAR-10 controls in Table 2, with three seeds each (69 runs).
- `classification/<id>/summary.json`: final training, validation and test results, calibration count, data-split hashes, runtime and peak GPU memory.
- `classification/<id>/epochs.jsonl`: all 100 epoch records, including training and internal-validation metrics. Test metrics are recorded only at final evaluation. Runtime fields are recorded wall-clock values from the original execution, including shared-GPU runs; they are not isolated speed benchmarks.
- `routing/`: six GrAPE model records (uncalibrated and calibration every 10 epochs, three seeds), with a combined `summary.json` and per-seed NumPy arrays. These use the first 256 internal-validation images in batches of 16, measuring all four receiving blocks. The arrays include per-image measurements and the fixed CIFAR-10 examples used for visualization.
- `credit_maps/fixed_error.npz`: BP, GrAPE and FOCF credit maps for the same four images and fixed output contrast, using each method's own seed-0 model. The JSON records indices, normalization, model configurations and frozen-state checks.
- `route_swap/`: one record per seed, containing held-out response errors for the identity carrier, matched correction and exchanged correction, plus the three-seed summary. All 1,000 internal-validation images are evaluated in batches of 32 over the first three boundaries. Correction exchange preserves the recipient's carrier and correction norm.
- `output_budget/`: six records (two datasets, three seeds), containing per-image response errors, acceptance decisions and fit timings for 4, 8, 16, 32 and full-class output codes. Each uses 256 internal-validation images in batches of 16 and three timing repetitions. The summary aggregates seed-level measurements for Table 3.

Load array files with `numpy.load(path, allow_pickle=False)`; field names describe the measured quantity. Checkpoint and frozen-state hashes identify the original measurement state. Original workstation paths and private execution metadata have been removed. Reproduction configurations are in `configs/training/`, `configs/diagnostic_training/` and `configs/diagnostics/`.
