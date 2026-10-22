#!/usr/bin/env python3
"""Plot the frozen FOCF correction-swap measurements in Figure S1."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "data/route_swap/summary.json").read_text())["by_boundary"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for method, label, color in [("matched", "Matched correction", "#2F6F91"),
                                  ("swapped", "Exchanged correction", "#B77B48")]:
        ax.errorbar(np.arange(1, 4), [data[str(b)][method]["mean"] for b in range(3)],
                    yerr=[data[str(b)][method]["sample_sd"] for b in range(3)],
                    label=label, color=color, marker="o", capsize=3)
    ax.axhline(1, color="black", linestyle=":", label="Identity carrier")
    ax.set(xlabel="Receiving block", ylabel="Held-out relative response MSE", xticks=[1, 2, 3], ylim=(0, 1.08))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = ROOT / "outputs/figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure_S1.pdf")
    fig.savefig(out / "figure_S1.png", dpi=170)
    plt.close(fig)
    print("Wrote outputs/figures/figure_S1.pdf")


if __name__ == "__main__":
    main()
