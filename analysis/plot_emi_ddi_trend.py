# plot fig1

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#plot settings
matplotlib.rcParams.update({'font.size': 18})
plt.rc("axes.spines", top=False, right=True)
matplotlib.rcParams['lines.markersize'] = 5
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams["legend.handlelength"] = 2.0
LABELSPACING = 0.6

EMI_BLUE = "#0015BC"
DDI_RED  = "#D62728"

# Country registry 
# df entry can be either:
# - a single DataFrame
# - a list of (segment_label, df_segment) tuples
boot_df_de_od_lt90 = pd.read_csv('../aggregate_data/DE/bootstrapyearly_sbertllms_lt1990.csv')
boot_df_de_od_ge90 = pd.read_csv('../aggregate_data/DE/bootstrapyearly_sbertllms_ge1990.csv')
boot_df_it_attributed = pd.read_csv('../aggregate_data/IT/bootstrapyearly_sbertllms_gt1945.csv')
boot_df_is = pd.read_csv('../aggregate_data/IS/bootstrapyearly_sbertllms_gt1945.csv')
boot_df = pd.read_csv('../aggregate_data/US/bootstrapyearly_sbertllms_gt1945.csv')
boot_df_tr_attributed = pd.read_csv('../aggregate_data/TR/bootstrapyearly_sbertllms_gt1981.csv')
boot_df_pl = pd.read_csv('../aggregate_data/PL/bootstrapyearly_sbertllms_gt1989.csv')
registry = [
    ("A) Germany", "DE", [("FRG", boot_df_de_od_lt90), ("DEU", boot_df_de_od_ge90)], 1945),
    ("B) Italy",   "IT", boot_df_it_attributed, 1945),
    ("C) Iceland", "IS", boot_df_is,            1945),
    ("D) U.S.",    "US", boot_df,               1945),
    ("E) Turkey",  "TR", boot_df_tr_attributed, 1981),
    ("F) Poland",  "PL", boot_df_pl,            1989),
]

# Column names for the long-format bootstrap df
COLS = dict(
    year="year",
    score="score",
    mean="mean",
    ci_lo="ci_lower",
    ci_hi="ci_upper",
    ddi="ddi",
)

SCORE_LABELS = {
    "ev_pool_z": "Evidence",
    "int_pool_z": "Intuition",
    "rating_emb_pool_z": "EMI"
}

def score_label(score_name: str) -> str:
    try:
        return SCORE_LABELS[score_name]
    except KeyError:
        raise KeyError(
            f"No label defined for score='{score_name}'. "
            f"Add it to SCORE_LABELS."
        )

#score series to plot on the left axis
TARGET_SCORE = "rating_emb_pool_z"

# helpers
def _iter_segments(df_or_segments):
    """
    Yields (segment_label, df_segment) pairs.
    If a plain DataFrame is passed, yields ('ALL', df).
    """
    if isinstance(df_or_segments, list):
        for seg_label, seg_df in df_or_segments:
            yield seg_label, seg_df
    else:
        yield "ALL", df_or_segments

def plot_score_with_ci(ax, df_or_segments, score_name: str):
    label = score_label(score_name)
    first = True

    for _, df_long in _iter_segments(df_or_segments):
        d = df_long[df_long[COLS["score"]] == score_name].copy()
        d = d.sort_values(COLS["year"])
        if d.empty:
            continue

        x = d[COLS["year"]].to_numpy(dtype=float)
        y = d[COLS["mean"]].to_numpy(dtype=float)
        lo = d[COLS["ci_lo"]].to_numpy(dtype=float)
        hi = d[COLS["ci_hi"]].to_numpy(dtype=float)

        ax.fill_between(x, lo, hi, color=EMI_BLUE, alpha=0.18, linewidth=0)
        ax.plot(
            x, y,
            color=EMI_BLUE,
            marker="o",
            linewidth=0.5,
            label=(label if first else None)
        )
        first = False

def plot_ddi(ax, df_or_segments):
    first = True

    for _, df_long in _iter_segments(df_or_segments):
        d = (
            df_long[[COLS["year"], COLS["ddi"]]]
            .dropna()
            .drop_duplicates(subset=[COLS["year"]])
            .sort_values(COLS["year"])
        )
        if d.empty:
            continue

        x = d[COLS["year"]].to_numpy(dtype=float)
        y = d[COLS["ddi"]].to_numpy(dtype=float)

        ax.plot(
            x, y,
            color=DDI_RED,
            marker="s",
            label=("DDI" if first else None)
        )
        first = False

def _segment_year_minmax(df_or_segments):
    mins, maxs = [], []
    for _, d in _iter_segments(df_or_segments):
        mins.append(d[COLS["year"]].min())
        maxs.append(d[COLS["year"]].max())
    return min(mins), max(maxs)

# filter and global x-axis
filtered = []

for title, code, df_or_segments, cutoff in registry:
    kept_segments = []

    for seg_label, seg_df in _iter_segments(df_or_segments):
        d = seg_df.copy()
        d = d[d[COLS["year"]] > cutoff].dropna(
            subset=[
                COLS["year"],
                COLS["score"],
                COLS["mean"],
                COLS["ci_lo"],
                COLS["ci_hi"],
            ]
        )

        d_score = d[d[COLS["score"]] == TARGET_SCORE]
        if d_score.empty:
            raise ValueError(
                f"{code} ({seg_label}): TARGET_SCORE='{TARGET_SCORE}' not found in "
                f"df['{COLS['score']}']."
            )

        kept_segments.append((seg_label, d))

    # Keep Germany as list of segments; others as plain df
    if isinstance(df_or_segments, list):
        filtered.append((title, code, kept_segments))
    else:
        filtered.append((title, code, kept_segments[0][1]))

global_min = min(_segment_year_minmax(d)[0] for _, _, d in filtered)
global_max = max(_segment_year_minmax(d)[1] for _, _, d in filtered)

EVENTS = {
    "DE": {"year": 1990, "label": "Unification (1990)"},
    "TR": {"year": 1982, "label": "1982 constitution"},
    "PL": {"year": 1990, "label": "Post-communist rule"},
}
# Figure
fig, axes = plt.subplots(
    nrows=len(filtered),
    ncols=1,
    sharex=True,
    figsize=(15, 25)
)

if len(filtered) == 1:
    axes = [axes]

for ax, (title, code, d) in zip(axes, filtered):
    ax.grid(True, axis='x')
    ax.grid(False, axis='y')
    ax.set_title(title, fontweight='bold', loc='left', x=0.01, y=0.95)

    # Left axis: score series + CI
    plot_score_with_ci(ax, d, TARGET_SCORE)
    ax.set_ylabel(score_label(TARGET_SCORE), fontweight='bold')

    # Right axis: DDI
    ax2 = ax.twinx()
    plot_ddi(ax2, d)
    ax2.set_ylabel("DDI", fontweight='bold')

    # Germany: indicate discontinuity / unification
    EVENT_COLOR = "black" 

    if code in EVENTS:
        event_year = EVENTS[code]["year"]
        event_label = EVENTS[code]["label"]
    
        ax.axvline(
            event_year,
            color=EVENT_COLOR,
            linestyle="--",
            linewidth=1.75,
            alpha=0.7,
            zorder=0
        )
    
        ax.annotate(
            event_label,
            xy=(event_year, 0.85),
            xycoords=ax.get_xaxis_transform(),
            xytext=(event_year + 3, 1.02),
            textcoords=ax.get_xaxis_transform(),
            arrowprops=dict(
                arrowstyle="-|>",
                color=EVENT_COLOR,
                lw=1.2,
                alpha=0.8
            ),
            ha="right",
            va="bottom",
            # fontsize=11,
            color=EVENT_COLOR
        )

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc='lower left',
        bbox_to_anchor=(0.01, 0.65),
        frameon=False,
        handlelength=1.5,
        labelspacing=LABELSPACING,
        handletextpad=0.75
    )

axes[-1].set_xlabel("Year", fontweight='bold')

for ax in axes:
    ax.set_xlim(global_min - 1, global_max + 1)

plt.tight_layout()

plt.savefig('./output/emi_dditrend.pdf', format='pdf', dpi=300)
#plt.savefig('./output/emi_dditrend.png', format='png', dpi=300)
#plt.savefig('./output/emi_dditrend.svg', format='svg', dpi=300)

plt.show()
