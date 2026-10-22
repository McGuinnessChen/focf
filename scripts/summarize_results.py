#!/usr/bin/env python3
"""Rebuild the numerical tables from the released measurements."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables"


def stats(values):
    return statistics.mean(values), statistics.stdev(values)


def save(name, rows):
    with (OUTPUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = list(csv.DictReader((ROOT / "data/classification/per_seed.csv").open()))
    assert len(records) == 69
    groups = {}
    for row in records:
        groups.setdefault((row["dataset"], row["method"]), {})[int(row["seed"])] = float(row["test_accuracy_percent"])
    rows = []
    for (dataset, method), seeds in sorted(groups.items()):
        assert set(seeds) == {0, 1, 2}
        if method in ("identity", "random_dictionary"):
            continue
        mean, sd = stats(list(seeds.values()))
        rows.append(dict(dataset=dataset, method=method, mean_percent=mean, sample_sd_percent=sd))
    save("table_1.csv", rows)
    rows = []
    focf = groups[("cifar10", "focf")]
    for method in ("focf", "random_dictionary", "identity"):
        values = groups[("cifar10", method)]
        mean, sd = stats(list(values.values()))
        gain, gain_sd = stats([focf[s] - values[s] for s in range(3)])
        rows.append(dict(method=method, mean_percent=mean, sample_sd_percent=sd,
                         paired_gain_pp=gain, paired_gain_sample_sd_pp=gain_sd,
                         suffix_queries_per_sample=0 if method == "identity" else 1788))
    save("table_2.csv", rows)
    data = json.loads((ROOT / "data/output_budget/summary.json").read_text())
    rows = []
    for dataset, budgets in data["datasets"].items():
        for budget, measurements in budgets.items():
            row = dict(dataset=dataset, output_codes=budget)
            for key in ("relative_response_mse", "relative_fit_accept_time", "acceptance_fraction", "retained_error_reduction"):
                values = measurements[key]["per_seed"]
                mean, sd = stats(values)
                assert abs(mean - measurements[key]["mean"]) < 1e-12
                assert abs(sd - measurements[key]["sample_sd"]) < 1e-12
                row[key + "_mean"] = mean
                row[key + "_sample_sd"] = sd
            rows.append(row)
    save("table_3.csv", rows)
    data = json.loads((ROOT / "data/routing/summary.json").read_text())["results"]
    rows = []
    for method in ("grape", "grape_T10"):
        seeds = [r for r in data if r["method"] == method]
        assert len(seeds) == 3
        for boundary in range(4):
            row = dict(method=method, block=boundary + 1)
            for key in ("input_variation_energy_fraction", "unit_operator_directional_variation", "grape_effect", "exchanged_effect"):
                mean, sd = stats([r["records"][boundary][key] for r in seeds])
                scale = 100 if "variation" in key else 1
                row[key + "_mean"] = scale * mean
                row[key + "_sample_sd"] = scale * sd
            rows.append(row)
    save("table_S1.csv", rows)
    print("Wrote Tables 1, 2, 3 and S1 to outputs/tables/ (mean and sample SD over three seeds).")


if __name__ == "__main__":
    main()
