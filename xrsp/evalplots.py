"""Publication figures for the unified model's held-out evaluation.

Pure matplotlib -- seaborn is not in the container, and a plotting dependency is a poor
reason to rebuild a 14 GB image.

Every figure is written as BOTH .pdf and .png: the PDF is vector, which is what a journal
wants and what survives being scaled into a two-column layout; the PNG is for pasting into
a slide or an issue. Fonts are embedded as TrueType (Type 42) because Type 3 -- the
matplotlib default -- is rejected outright by several submission systems.

Nothing here computes a statistic. Every number displayed arrives from evalmetrics, so a
figure and the JSON summary cannot drift apart.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

# Colour choices are deliberate, not decorative: this palette is distinguishable under
# the common forms of colour blindness AND separates in greyscale, because a reviewer
# may well print the paper. Reds are reserved for limits/failures so they never merely
# mean "series 2".
C_MAIN, C_ALT, C_WARN, C_GREY = "#0072B2", "#009E73", "#D55E00", "#666666"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")               # no display on a compute node
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,       # TrueType, not Type 3
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.bbox": "tight",
    })
    return plt


def _save(fig, out_dir: str, stem: str) -> list:
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=300)
        paths.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def plot_ced(curves: Dict[str, tuple], out_dir: str, stem: str = "fig_ced",
             xlabel: str = "error threshold (px)"):
    """Cumulative error distribution, one line per group.

    Reference lines at the thresholds people actually quote, so a reader can read off
    "what fraction is within 5 px" without a ruler.
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for i, (label, (th, frac)) in enumerate(curves.items()):
        ax.plot(th, 100 * np.asarray(frac), lw=1.6,
                color=[C_MAIN, C_ALT, C_WARN, C_GREY][i % 4], label=label)
    for x in (2.0, 5.0):
        ax.axvline(x, color=C_GREY, lw=0.6, ls=":", zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("landmarks within threshold (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, out_dir, stem)


def plot_error_by_level(err_by_level: Dict[str, Sequence[float]], out_dir: str,
                        stem: str = "fig_error_by_level"):
    """Per-level error distribution, cranial->caudal.

    A box plot rather than a bar of means: landmark error is right-skewed and a bar chart
    of means hides exactly the level that fails occasionally and badly. n is printed under
    each box because levels differ enormously in how often they are in the field of view,
    and a tight box over four observations is not a result.
    """
    plt = _mpl()
    levels = [lv for lv in err_by_level
              if len([v for v in err_by_level[lv] if np.isfinite(v)]) > 0]
    data = [[v for v in err_by_level[lv] if np.isfinite(v)] for lv in levels]
    fig, ax = plt.subplots(figsize=(max(3.4, 0.42 * len(levels) + 1), 2.9))
    bp = ax.boxplot(data, labels=levels, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.2), widths=0.62)
    for patch in bp["boxes"]:
        patch.set(facecolor=C_MAIN, alpha=0.45, lw=0.8)
    for lv, d, x in zip(levels, data, range(1, len(levels) + 1)):
        ax.annotate(f"{len(d)}", (x, 0), xytext=(0, -22), textcoords="offset points",
                    ha="center", fontsize=6, color=C_GREY)
    ax.set_ylabel("radial error (px)")
    ax.set_xlabel("level  (n below)")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, out_dir, stem)


def plot_bland_altman(ba_by_param: Dict[str, Dict], out_dir: str,
                      stem: str = "fig_bland_altman", unit: str = "deg"):
    """Bland-Altman per spinopelvic parameter -- the agreement figure for a new method.

    Bias and 95% limits are drawn and annotated. The limits, not the bias, are the result:
    a method can be unbiased on average and still be +-15 deg on any given patient, and
    only this plot shows that.
    """
    plt = _mpl()
    params = [p for p in ba_by_param if ba_by_param[p].get("n", 0) >= 2]
    if not params:
        return []
    n = len(params)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.6 * nrow), squeeze=False)
    for ax, p in zip(axes.ravel(), params):
        d = ba_by_param[p]
        ax.scatter(d["mean"], d["diff"], s=9, alpha=0.5, color=C_MAIN,
                   edgecolors="none")
        ax.axhline(d["bias"], color=C_ALT, lw=1.2)
        ax.axhline(d["loa_low"], color=C_WARN, lw=1.0, ls="--")
        ax.axhline(d["loa_high"], color=C_WARN, lw=1.0, ls="--")
        ax.axhline(0, color=C_GREY, lw=0.6, zorder=0)
        ax.set_title(f"{p}  (n={d['n']})")
        ax.set_xlabel(f"mean of methods ({unit})")
        ax.set_ylabel(f"predicted - truth ({unit})")
        ax.annotate(f"bias {d['bias']:+.1f}\nLoA {d['loa_low']:+.1f}, {d['loa_high']:+.1f}",
                    xy=(0.02, 0.03), xycoords="axes fraction", fontsize=7, color=C_GREY)
        ax.grid(alpha=0.22, lw=0.5)
    for ax in axes.ravel()[len(params):]:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, out_dir, stem)


def plot_scatter_agreement(pairs: Dict[str, tuple], stats: Dict[str, Dict],
                           out_dir: str, stem: str = "fig_agreement",
                           unit: str = "deg"):
    """Predicted vs true with the LINE OF IDENTITY, not a fitted regression line.

    A fitted line is the more flattering choice and the wrong one: it shows how well the
    two correlate after absorbing any bias into its intercept. y=x is the claim actually
    being made -- that the predicted value IS the measurement.
    """
    plt = _mpl()
    params = [p for p in pairs if len(pairs[p][0]) >= 2]
    if not params:
        return []
    ncol = 2 if len(params) > 1 else 1
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.9 * nrow), squeeze=False)
    for ax, p in zip(axes.ravel(), params):
        t, q = np.asarray(pairs[p][0], float), np.asarray(pairs[p][1], float)
        m = np.isfinite(t) & np.isfinite(q)
        t, q = t[m], q[m]
        lo, hi = float(min(t.min(), q.min())), float(max(t.max(), q.max()))
        pad = 0.05 * (hi - lo + 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=C_GREY, lw=0.9, ls="--")
        ax.scatter(t, q, s=10, alpha=0.55, color=C_MAIN, edgecolors="none")
        s = stats.get(p, {})
        ax.annotate(f"ICC {s.get('icc', float('nan')):.2f}\n"
                    f"r {s.get('pearson_r', float('nan')):.2f}\n"
                    f"MAE {s.get('mae', float('nan')):.1f}{unit}",
                    xy=(0.03, 0.72), xycoords="axes fraction", fontsize=7)
        ax.set_title(f"{p}  (n={t.size})")
        ax.set_xlabel(f"ground truth ({unit})")
        ax.set_ylabel(f"predicted ({unit})")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.22, lw=0.5)
    for ax in axes.ravel()[len(params):]:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, out_dir, stem)


def plot_confusion(M: np.ndarray, classes: Sequence[str], out_dir: str, stem: str,
                   title: str = "", prf: Optional[Dict] = None, dropped: int = 0):
    """A confusion matrix showing BOTH the row-normalised colour and the raw count.

    Row-normalised because the classes are unbalanced and a count-coloured matrix just
    renders the class prior. The raw n stays in the cell so a convincing 100% over three
    cases cannot masquerade as a result. Per-class F1 runs down the right-hand side.
    """
    plt = _mpl()
    M = np.asarray(M, float)
    k = len(classes)
    rows = M.sum(1, keepdims=True)
    norm = np.divide(M, rows, out=np.zeros_like(M), where=rows > 0)
    fig, ax = plt.subplots(figsize=(0.72 * k + 2.4, 0.72 * k + 1.9))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xticks(range(k)); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(k)); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
    if prf:
        for i in range(k):
            ax.annotate(f"F1 {prf['f1'][i]:.2f}", xy=(1.02, 1 - (i + 0.5) / k),
                        xycoords="axes fraction", fontsize=7, va="center", color=C_GREY)
        sub = (f"accuracy {prf['accuracy']:.2f}   macro-F1 {prf['macro_f1']:.2f}   "
               f"kappa {prf['kappa']:.2f}")
        if dropped:
            sub += f"\n{dropped} unclassifiable, excluded"
        ax.set_title((title + "\n" if title else "") + sub, fontsize=8)
    elif title:
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.16, label="row fraction")
    fig.tight_layout()
    return _save(fig, out_dir, stem)


def plot_identity_residual(pi: Sequence[float], ss: Sequence[float],
                           pt: Sequence[float], out_dir: str,
                           stem: str = "fig_pi_identity"):
    """PI - (SS + PT), which is identically zero by construction.

    Not a measure of accuracy against ground truth -- it is a measure of INTERNAL
    CONSISTENCY, and it needs no labels at all. The three parameters are derived from two
    predicted structures (the S1 endplate and the hip axis), so any spread here is the
    model disagreeing with itself. That makes it the one quality signal that also works
    on unlabelled clinical data, where nothing else on this page can be computed.
    """
    plt = _mpl()
    r = np.asarray(pi, float) - (np.asarray(ss, float) + np.asarray(pt, float))
    r = r[np.isfinite(r)]
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    if r.size:
        ax.hist(r, bins=min(40, max(8, r.size // 5)), color=C_MAIN, alpha=0.8)
        ax.axvline(0, color=C_WARN, lw=1.1, ls="--")
        ax.annotate(f"n {r.size}\nmedian {np.median(r):+.2f}\n"
                    f"p95 |r| {np.percentile(np.abs(r), 95):.2f}",
                    xy=(0.02, 0.68), xycoords="axes fraction", fontsize=7)
    ax.set_xlabel("PI - (SS + PT)  (deg)")
    ax.set_ylabel("cases")
    ax.grid(alpha=0.22, lw=0.5)
    return _save(fig, out_dir, stem)
