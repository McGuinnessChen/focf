#!/usr/bin/env python3
"""Build method credit maps and the unchanged GrAPE routing diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/focf_mpl_config")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data/routing"
LATEX = HERE.parent / "outputs"
BLUE, ORANGE, GRAY = "#2F6F91", "#B77B48", "#000000"


def stats(results, method, boundary, key):
    values = [r["records"][boundary][key] for r in results if r["method"] == method]
    assert len(values) == 3
    return float(np.mean(values)), float(np.std(values, ddof=1))


def axis_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#EAEAEA", linewidth=.6)
    ax.set_axisbelow(True)
    ax.set_xticks([1, 2, 3, 4], ["1", "2", "3", "4*"])
    ax.tick_params(length=3, width=.8, labelsize=12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-table", action="store_true",
                        help="Explicitly regenerate Table S1 from the original diagnostics")
    args = parser.parse_args()
    (LATEX / "figures").mkdir(parents=True, exist_ok=True)
    (LATEX / "tables").mkdir(parents=True, exist_ok=True)
    source = json.loads((DATA / "summary.json").read_text())
    results = source["results"]
    assert len(results) == 6
    assert all(r["frozen_digest_before"] == r["frozen_digest_after"] for r in results)
    assert all(r["parameter_updates"] == r["feedback_updates"] == 0 for r in results)
    assert all(not r["test_accessed"] for r in results)
    example = next(r for r in results if r["method"] == "grape" and r["seed"] == 0)
    prefix = HERE.parent / "data/credit_maps/fixed_error"
    method_record = json.loads(prefix.with_suffix(".json").read_text())
    assert method_record["array_sha256"] == hashlib.sha256(prefix.with_suffix(".npz").read_bytes()).hexdigest()
    arrays = np.load(prefix.with_suffix(".npz"))
    methods = ("bp", "grape", "focf")
    magnitude = [arrays[f"{name}_patch_magnitude"] for name in methods]
    cls = [arrays[f"{name}_cls_fraction"] for name in methods]
    for name in methods:
        assert method_record["methods"][name]["frozen_before"] == method_record["methods"][name]["frozen_after"]
        assert method_record["methods"][name]["checkpoint_epoch"] == 100
    np.testing.assert_allclose(arrays["grape_unit_credit"],
                               np.repeat(arrays["grape_unit_credit"][:1], 4, axis=0), atol=0)
    vmax = max(float(m.max()) for m in magnitude)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.labelsize": 12, "axes.titlesize": 13,
                         "text.color": "black", "axes.labelcolor": "black",
                         "xtick.color": "black", "ytick.color": "black",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    # Lay out in inches: shared title baselines, square image cells, and a
    # balanced lower edges for the colorbar and C's axis label. Font sizes
    # stay legible in the paper.
    width, height = 10.4, 7.2
    fig = plt.figure(figsize=(width, height), facecolor="white")

    def axes(x, y, w, h):
        return fig.add_axes([x / width, y / height, w / width, h / height])

    def panel_title(letter, title, x, y):
        fig.text(x / width, y / height, letter, fontsize=14, weight="bold", va="baseline")
        fig.text((x + .29) / width, y / height, title,
                 fontsize=14, weight="bold", va="baseline")

    panel_title("A", "Same error, different inputs", .12, 6.92)
    panel_title("B", "Variation across inputs", 5.95, 6.92)
    panel_title("C", "Local loss reduction", 5.95, 3.43)
    for row in range(4):
        for col in range(4):
            ax = axes(.65 + col * 1.07, [5.56, 4.14, 2.72, 1.30][row], .98, .98)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.imshow(arrays["images"][col], interpolation="nearest")
                ax.set_title(method_record["classes"][col].capitalize(), fontsize=12, pad=7)
            else:
                mappable = ax.imshow(magnitude[row - 1][col], cmap="Blues", vmin=0, vmax=vmax,
                                    interpolation="nearest")
                ax.set_xlabel(f"CLS {100 * cls[row - 1][col]:.0f}%", fontsize=11, labelpad=6)
            if col == 0:
                ax.set_ylabel(["Input", "BP", "GrAPE", "FOCF"][row],
                              fontsize=12, rotation=90, labelpad=10)
    color_ax = axes(1.20, .43, 3.10, .12)
    bar = fig.colorbar(mappable, cax=color_ax, orientation="horizontal")
    bar.set_ticks([0, .1, .2, .3])
    bar.set_label("Patch-credit magnitude", fontsize=12, labelpad=4)
    bar.ax.tick_params(labelsize=11, length=2)
    bar.outline.set_visible(False)
    ax_b = axes(5.95, 4.05, 4.20, 2.50)
    values = np.array([stats(results, "grape", b, "input_variation_energy_fraction")
                       for b in range(4)])
    ax_b.bar(np.arange(1, 5), 100*values[:, 0], yerr=100*values[:, 1], width=.55,
             color="#89AFC4", edgecolor=BLUE, linewidth=.8,
             error_kw={"ecolor": GRAY, "elinewidth": 1.2, "capsize": 3, "capthick": 1.2})
    ax_b.set_ylabel("Uncaptured variation (%)", labelpad=8)
    ax_b.set_yticks([0, 20, 40, 60, 80])
    ax_b.set_ylim(0, 80)
    ax_b.set_xlim(.45, 4.55)
    axis_style(ax_b)
    # The stacked plots share the receiving-block coordinate. Label it once
    # below C to leave a clear gap between B's ticks and C's heading.
    ax_c = axes(5.95, .56, 4.20, 2.55)
    legend_handles, legend_labels = [], []
    for key, label, color in [("grape_effect", "GrAPE", BLUE),
                               ("exchanged_effect", "Exact credit (exchanged)", ORANGE)]:
        values = [stats(results, "grape", b, key) for b in range(4)]
        curve = ax_c.errorbar(np.arange(1, 5), np.array(values)[:, 0], yerr=np.array(values)[:, 1],
            color=color, linestyle="-", marker="o", markerfacecolor="white",
            markersize=6, linewidth=1.7, capsize=3)
        # Legend handles match the plotted line and hollow-circle marker.
        legend_handles.append(curve.lines[0])
        legend_labels.append(label)
    ax_c.axhline(1, color=GRAY, linewidth=1, linestyle=":")
    ax_c.set_ylabel("Relative to exact credit", labelpad=8)
    ax_c.set_yticks([0, .5, 1])
    ax_c.set_ylim(0, 1.15)
    ax_c.set_xlim(.45, 4.55)
    ax_c.set_xlabel("Receiving block", labelpad=6)
    axis_style(ax_c)
    ax_c.text(.65, 1.035, "Exact credit (current input)", fontsize=10.5, va="bottom")
    ax_c.legend(handles=legend_handles, labels=legend_labels,
                frameon=False, fontsize=10.5, loc="center right", bbox_to_anchor=(.99, .49),
                borderaxespad=.2, labelspacing=.45)
    output = LATEX / "figures/figure_2_routing_bottleneck"
    fig.savefig(str(output) + ".pdf", bbox_inches="tight", pad_inches=.08)
    fig.savefig(str(output) + ".svg", bbox_inches="tight", pad_inches=.08)
    fig.savefig(LATEX / "figures/figure_2_routing_bottleneck.png", dpi=170, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)
    lines = [r"\begin{table}[p]", r"\centering", r"\small",
        r"\caption{Input-dependent downstream routing in frozen GrAPE baselines.}",
        r"\label{tab:S1}",
        r"\begin{tabularx}{\textwidth}{YcYYYY}", r"\toprule",
        r"\textbf{Calibration} & \textbf{Block} & \textbf{Input variation (\%)} & \textbf{Unit-norm variation (\%)} & \textbf{GrAPE loss effect} & \textbf{Exchanged-route loss effect} \\",
        r"\midrule"]
    for method, label in [("grape", "None"), ("grape_T10", r"Every 10 epochs")]:
        for b in range(4):
            cells = [label if b == 0 else "", str(b + 1) + (r"$^*$" if b == 3 else "")]
            for key in ["input_variation_energy_fraction", "unit_operator_directional_variation", "grape_effect", "exchanged_effect"]:
                mean, sd = stats(results, method, b, key)
                scale = 100 if "variation" in key else 1
                decimals = 2 if scale == 100 else 3
                cells.append(r"\(" + f"{scale*mean:.{decimals}f}" + r"\pm" + f"{scale*sd:.{decimals}f}" + r"\)")
            lines.append(" & ".join(cells) + r" \\")
        if method == "grape":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\par\smallskip",
        r"{\footnotesize Mean $\pm$ sample SD over seeds 0, 1, and 2. Each epoch-100 CIFAR-10 checkpoint uses the same 256 internal-validation images. $^*$Block 4 has only the final normalization and classifier downstream. Input variation is the squared deviation of centered suffix Jacobians from their cohort mean, divided by their total squared norm. Unit-norm variation repeats the calculation after normalizing each sample operator, separating directional variation from magnitude. Loss effects use equal-norm hidden-state displacements, measured by central forward differences and normalized by the current-input exact directional derivative (reference $\simeq1$). Exchanged operators come from the preceding sample cyclically within each fixed batch of 16 and act on the recipient's own output error. These are frozen-model diagnostics, not accuracy differences or optimizer-update measurements.}",
        r"\end{table}"]
    if args.refresh_table:
        (LATEX / "tables/table_S1.tex").write_text("\n".join(lines) + "\n")
    record = {"source": str(DATA / "summary.json"), "source_sha256": hashlib.sha256((DATA / "summary.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "figure_pdf_sha256": hashlib.sha256(Path(str(output) + ".pdf").read_bytes()).hexdigest(),
        "selection": example["example_selection"], "output_contrast": "Truck logit minus mean logit; same vector for all methods and inputs",
        "panel_A_source": str(prefix.with_suffix(".json")),
        "panel_A_source_sha256": hashlib.sha256(prefix.with_suffix(".json").read_bytes()).hexdigest(),
        "panel_A_methods": ["bp", "grape", "focf"], "panel_A_model_rule": "Each method uses its own epoch-100 seed-0 trained model and feedback",
        "panel_A_normalization": "Unit norm of whole credit including CLS; one common color scale",
        "panels_B_C_methods": ["grape"], "panels_B_C_measurements_unchanged": True,
        "supplementary_methods": ["grape", "grape_T10"],
        "scope": "A: four fixed examples; B/C: original 256-image three-seed GrAPE measurements; no model updates"}
    (LATEX / "figure_2_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Built Figure 2: BP/GrAPE/FOCF maps with original GrAPE diagnostics.")


if __name__ == "__main__":
    main()
