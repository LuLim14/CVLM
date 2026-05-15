"""Generate diploma figures from CVLM eval JSONs.

Reads `eval_cvlm_native_step*.json` from the configured run directories
(d1/d3/d4) and writes four figures into `text_diploma/graphics/`, each in
both `.pdf` (vector, for the LaTeX build) and `.png` (raster, for slides /
quick preview):

  1. cr_progression_k1_vs_k4  — three panels (PPL/R-L/TokAcc) vs cr.
  2. training_curves_K4       — three panels over training steps for d4
                                with curriculum stages shaded.
  3. progression_summary      — bar chart of the cumulative improvement chain
                                at cr=8 (mean-pool → attn → unfreeze → +K=4).
  4. pareto_cr_rougeL         — single panel: cr (log2) vs ROUGE-L for
                                K=1 vs K=4 plus baseline horizontals.
  5. compression_efficiency.tex — LaTeX table with BPST/ECR from d4 JSONs.

Usage (with the conda env that has matplotlib 3.10):
  PYTHONNOUSERSITE=1 python text_diploma/scripts/make_plots.py
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CKPT_ROOT = "/home/jovyan/shares/SR008.fs2/gigachat_checkpoints/rl/ckpts/MoE-losses/cvlm"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "graphics")

RUNS = {
    "d1":      "run_20260508_001834_cr1_vl1024_sl0_bs64_lr1e-4_ep8_gc1_unfrzed22",
    "d3":      "run_20260510_201431_cr1_vl1024_sl0_bs64_lr1e-4_ep10_gc1_unfrzed22",
    "d4":      "run_20260510_212404_cr1_vl1024_sl0_bs64_lr1e-4_ep10_gc1_unfrzed22_lt4",
    "frozen_attn":     "run_20260507_003517_cr1_vl1024_sl0_bs64_lr1e-4_ep8_gc1",
    "frozen_meanpool": "run_20260506_011615_cr1_vl1024_sl0_bs64_lr1e-4_ep8_gc1",
}

BASELINE_RL_LOWER = 0.097
BASELINE_RL_UPPER = 0.165

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})

PNG_DPI = 200  # raster resolution for .png companions


def save_fig(fig, base_path_no_ext: str) -> None:
    """Save the figure as both .pdf (vector) and .png (raster, dpi=PNG_DPI)."""
    for ext in (".pdf", ".png"):
        out = base_path_no_ext + ext
        fig.savefig(out, bbox_inches="tight", dpi=PNG_DPI if ext == ".png" else None)
        print(f"wrote {out}")
    plt.close(fig)


def load_run(run_dir: str) -> List[Dict]:
    pattern = os.path.join(run_dir, "eval_cvlm_native_step*.json")
    out: List[Dict] = []
    for path in sorted(glob.glob(pattern), key=lambda p: int(re.search(r"step(\d+)", p).group(1))):
        if "_samples.json" in path:
            continue
        with open(path) as f:
            j = json.load(f)
        step = int(re.search(r"step(\d+)", path).group(1))
        out.append({
            "step": step,
            "cr": int(j.get("compression_rate", 0)),
            "ppl": float(j["perplexity"]),
            "rL": float(j["rougeL"]),
            "tokacc": float(j["token_accuracy"]),
            "bleu4": float(j.get("bleu4", np.nan)),
            "bpst": float(j.get("bits_per_source_token", np.nan)),
            "ecr": float(j.get("effective_context_reduction", np.nan)),
            "cr_mean": float(j.get("compression_ratio_mean", np.nan)),
        })
    return out


def best_per_cr(rows: List[Dict], metric: str, higher_is_better: bool) -> Dict[int, Dict]:
    """For each compression rate, return the row with the best metric."""
    by_cr: Dict[int, Dict] = {}
    for r in rows:
        cr = r["cr"]
        if cr == 0:
            continue
        cur = by_cr.get(cr)
        if cur is None:
            by_cr[cr] = r
            continue
        if higher_is_better and r[metric] > cur[metric]:
            by_cr[cr] = r
        elif (not higher_is_better) and r[metric] < cur[metric]:
            by_cr[cr] = r
    return by_cr


# ---------------------------------------------------------------------------
# Plot 1: cr-progression, K=1 vs K=4 (3 panels)
# ---------------------------------------------------------------------------

def plot_cr_progression(d3: List[Dict], d4: List[Dict], out_base: str) -> None:
    # Use the best (last) checkpoint at each cr for each run.
    d3_ppl  = best_per_cr(d3, "ppl",    higher_is_better=False)
    d4_ppl  = best_per_cr(d4, "ppl",    higher_is_better=False)
    d3_rL   = best_per_cr(d3, "rL",     higher_is_better=True)
    d4_rL   = best_per_cr(d4, "rL",     higher_is_better=True)
    d3_ta   = best_per_cr(d3, "tokacc", higher_is_better=True)
    d4_ta   = best_per_cr(d4, "tokacc", higher_is_better=True)

    crs = sorted(set(d3_ppl) | set(d4_ppl))

    def col(d: Dict[int, Dict], key: str) -> List[float]:
        return [d[c][key] if c in d else np.nan for c in crs]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    panels = [
        ("PPL ↓",          col(d3_ppl, "ppl"),  col(d4_ppl, "ppl"),  False),
        ("ROUGE-L ↑",      col(d3_rL,  "rL"),   col(d4_rL,  "rL"),   True),
        ("Token accuracy ↑", col(d3_ta,  "tokacc"), col(d4_ta,  "tokacc"), True),
    ]
    for ax, (title, y_k1, y_k4, higher_better) in zip(axes, panels):
        ax.plot(crs, y_k1, marker="o", linewidth=2.0, label="K=1 (d3)", color="#1f77b4")
        ax.plot(crs, y_k4, marker="s", linewidth=2.0, label="K=4 (d4)", color="#d62728")
        if title == "ROUGE-L ↑":
            ax.axhline(BASELINE_RL_LOWER, ls="--", lw=1.0, color="gray",
                       label="baseline_llm")
            ax.axhline(BASELINE_RL_UPPER, ls=":",  lw=1.0, color="gray",
                       label="baseline_llm_full")
        ax.set_xlabel("compression rate (cr)")
        ax.set_ylabel(title)
        ax.set_xscale("log", base=2)
        ax.set_xticks(crs)
        ax.set_xticklabels([str(c) for c in crs])
        ax.legend(loc="best", framealpha=0.9)
        ax.set_title(title)

    fig.tight_layout()
    save_fig(fig, out_base)


# ---------------------------------------------------------------------------
# Plot 2: training curves for K=4 (3 panels over steps, stages shaded)
# ---------------------------------------------------------------------------

def plot_training_curves(rows: List[Dict], out_base: str) -> None:
    rows = sorted(rows, key=lambda r: r["step"])
    steps = [r["step"] for r in rows]
    ppls  = [r["ppl"]  for r in rows]
    rLs   = [r["rL"]   for r in rows]
    tas   = [r["tokacc"] for r in rows]
    crs   = [r["cr"]   for r in rows]

    # Curriculum boundaries (from EVAL_CR_SCHEDULE=1:6000,2:12000,4:18000,8:24000,16:0).
    stages = [(0, 6000, 1), (6000, 12000, 2), (12000, 18000, 4),
              (18000, 24000, 8), (24000, max(steps), 16)]
    stage_colors = {1: "#f0f0f0", 2: "#e0e8f0", 4: "#d0e0e8", 8: "#c8d8e8", 16: "#b8d0e0"}

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=True)
    series = [
        (axes[0], ppls, "PPL ↓"),
        (axes[1], rLs,  "ROUGE-L ↑"),
        (axes[2], tas,  "Token accuracy ↑"),
    ]
    for ax, y, title in series:
        for s_lo, s_hi, cr in stages:
            ax.axvspan(s_lo, s_hi, facecolor=stage_colors[cr], alpha=0.6, zorder=0)
        ax.plot(steps, y, marker="o", color="#d62728", linewidth=2.0, zorder=3)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

    # cr stage annotations on the top panel.
    ax_top = axes[0]
    ymax = ax_top.get_ylim()[1]
    for s_lo, s_hi, cr in stages:
        ax_top.text((s_lo + s_hi) / 2, ymax * 0.95,
                    f"cr={cr}", ha="center", va="top",
                    fontsize=9, fontweight="bold", color="#444")

    axes[-1].set_xlabel("шаг обучения")
    axes[0].set_title("Эволюция метрик по шагам обучения для K=4 (d4); фон — стадии куррикулума")

    fig.tight_layout()
    save_fig(fig, out_base)


# ---------------------------------------------------------------------------
# Plot 3: progression bars at cr=8 (mean-pool → attn → unfreeze → +K=4)
# ---------------------------------------------------------------------------

def _last_cr8(rows: List[Dict]) -> Dict:
    cr8 = [r for r in rows if r["cr"] == 8]
    if not cr8:
        return {}
    return sorted(cr8, key=lambda r: r["step"])[-1]


def plot_progression_bars(
    rows_mean: List[Dict],
    rows_attn: List[Dict],
    rows_unfrz: List[Dict],
    rows_k4: List[Dict],
    out_base: str,
) -> None:
    configs = [
        ("Mean-pool\n(frozen)",            _last_cr8(rows_mean)),
        ("+ Attention pool\nK=1 (frozen)", _last_cr8(rows_attn)),
        ("+ Unfreeze\nK=22 (K=1)",         _last_cr8(rows_unfrz)),
        ("+ Multi-query\nK=4",             _last_cr8(rows_k4)),
    ]
    labels = [c[0] for c in configs]
    ppls   = [c[1].get("ppl", np.nan)    for c in configs]
    rLs    = [c[1].get("rL",  np.nan)    for c in configs]
    tas    = [c[1].get("tokacc", np.nan) for c in configs]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    bar_color = ["#9ecae1", "#6baed6", "#3182bd", "#d62728"]
    x = np.arange(len(labels))

    for ax, vals, title, fmt in [
        (axes[0], ppls, "PPL ↓ при cr=8",        "{:.2f}"),
        (axes[1], rLs,  "ROUGE-L ↑ при cr=8",    "{:.3f}"),
        (axes[2], tas,  "Token accuracy ↑ при cr=8", "{:.3f}"),
    ]:
        bars = ax.bar(x, vals, color=bar_color, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title)
        # Annotate values on top of bars.
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=9)
        if title.startswith("ROUGE"):
            ax.axhline(BASELINE_RL_LOWER, ls="--", lw=1.0, color="gray",
                       label="baseline_llm (0.097)")
            ax.axhline(BASELINE_RL_UPPER, ls=":",  lw=1.0, color="gray",
                       label="baseline_llm_full (0.165)")
            ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, out_base)


# ---------------------------------------------------------------------------
# Plot 4: Pareto cr → ROUGE-L
# ---------------------------------------------------------------------------

def plot_pareto(d3: List[Dict], d4: List[Dict], out_base: str) -> None:
    d3_best = best_per_cr(d3, "rL", higher_is_better=True)
    d4_best = best_per_cr(d4, "rL", higher_is_better=True)
    crs = sorted(set(d3_best) | set(d4_best))
    y_k1 = [d3_best[c]["rL"] if c in d3_best else np.nan for c in crs]
    y_k4 = [d4_best[c]["rL"] if c in d4_best else np.nan for c in crs]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(crs, y_k1, marker="o", linewidth=2.0, markersize=8,
            label="K=1 (d3, baseline)", color="#1f77b4")
    ax.plot(crs, y_k4, marker="s", linewidth=2.0, markersize=8,
            label="K=4 (d4, multi-query)", color="#d62728")
    ax.axhline(BASELINE_RL_LOWER, ls="--", lw=1.2, color="gray",
               label="baseline_llm (нижняя)")
    ax.axhline(BASELINE_RL_UPPER, ls=":",  lw=1.2, color="gray",
               label="baseline_llm_full (оракул)")
    ax.set_xlabel("Степень сжатия (cr)")
    ax.set_ylabel("ROUGE-L")
    ax.set_xscale("log", base=2)
    ax.set_xticks(crs)
    ax.set_xticklabels([str(c) for c in crs])
    ax.set_title("Парето-кривая: качество vs сжатие")
    ax.legend(loc="lower left", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Shade the band between baselines.
    ax.fill_between(crs, BASELINE_RL_LOWER, BASELINE_RL_UPPER,
                    color="lightyellow", alpha=0.35, zorder=0)

    fig.tight_layout()
    save_fig(fig, out_base)


# ---------------------------------------------------------------------------
# Table: compression efficiency for K=4 (d4)
# ---------------------------------------------------------------------------

def write_compression_efficiency_table(rows: List[Dict], out_path: str) -> None:
    """Write a small LaTeX table for quality/efficiency metrics by cr."""
    best = best_per_cr(rows, "rL", higher_is_better=True)
    crs = sorted(best)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Качество и эффективность сжатия для лучшей конфигурации $K=4$ (d4). Чекпоинт выбирается по максимальному ROUGE-L на соответствующей стадии $\text{cr}$.}",
        r"\label{tab:compression-efficiency}",
        r"\footnotesize",
        r"\begin{tabular}{cccccccc}",
        r"\toprule",
        r"$\text{cr}$ & Шаг & PPL$\downarrow$ & R-L$\uparrow$ & TokAcc$\uparrow$ & $\overline{\mathrm{CR}}$ & BPST$\downarrow$ & ECR$\uparrow$ \\",
        r"\midrule",
    ]
    for cr in crs:
        row = best[cr]
        lines.append(
            f"{cr} & {row['step']} & {row['ppl']:.2f} & {row['rL']:.3f} & "
            f"{row['tokacc']:.3f} & {row['cr_mean']:.2f} & {row['bpst']:.3f} & {row['ecr']:.2f} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    loaded: Dict[str, List[Dict]] = {}
    for key, sub in RUNS.items():
        run_dir = os.path.join(CKPT_ROOT, sub)
        if not os.path.isdir(run_dir):
            print(f"[warn] run dir missing: {run_dir} (skip)")
            loaded[key] = []
            continue
        rows = load_run(run_dir)
        print(f"loaded {key}: {len(rows)} eval points from {sub}")
        loaded[key] = rows

    if loaded["d3"] and loaded["d4"]:
        plot_cr_progression(loaded["d3"], loaded["d4"],
                            os.path.join(OUT_DIR, "cr_progression_k1_vs_k4"))
        plot_pareto(loaded["d3"], loaded["d4"],
                    os.path.join(OUT_DIR, "pareto_cr_rougeL"))

    if loaded["d4"]:
        plot_training_curves(loaded["d4"],
                             os.path.join(OUT_DIR, "training_curves_K4"))
        write_compression_efficiency_table(
            loaded["d4"],
            os.path.join(OUT_DIR, "compression_efficiency.tex"),
        )

    if loaded["frozen_meanpool"] and loaded["frozen_attn"] and loaded["d1"] and loaded["d4"]:
        plot_progression_bars(
            loaded["frozen_meanpool"], loaded["frozen_attn"],
            loaded["d1"], loaded["d4"],
            os.path.join(OUT_DIR, "progression_summary"),
        )

    print(f"\ndone — figures in {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
