# plot count per year
# plot count of segments
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 18})
plt.rc("axes.spines", top=False, right=True)
matplotlib.rcParams['lines.markersize'] = 5
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams["legend.handlelength"] = 2.0

# data
tr_df_yearly_attributed = pd.read_csv('../aggregate_data/TR/agg_sbertllms_gt1981_attributed.csv')
is_df_yearly = pd.read_csv('../aggregate_data/IS/agg_sbertllms_gt1945.csv')
pl_df_yearly = pd.read_csv('../aggregate_data/PL/agg_sbertllms_gt1989.csv')
de_df_yearly_od_lt90 = pd.read_csv('../aggregate_data/DE/agg_sbertllms_lt1990.csv')
de_df_yearly_od_ge90 = pd.read_csv('../aggregate_data/DE/agg_sbertllms_ge1990.csv')
it_df_yearly_attributed = pd.read_csv('../aggregate_data/IT/agg_sbertllms_gt1945_attributed.csv')
us_df_sessions_attributed = pd.read_csv('../aggregate_data/US/agg_sbertllms_gt1945.csv')

us = us_df_sessions_attributed.copy()
us["source"] = "US"

it = it_df_yearly_attributed.copy()
it["source"] = "IT"

is_ = is_df_yearly.copy()
is_["source"] = "IS"

pl = pl_df_yearly.copy()
pl["source"] = "PL"

tr = tr_df_yearly_attributed.copy()
tr["source"] = "TR"

# Merge West Germany and unified Germany into one DE panel
de_west = de_df_yearly_od_lt90.copy()
de_post90 = de_df_yearly_od_ge90.copy()
de = pd.concat([de_west, de_post90], ignore_index=True)
de["source"] = "DE"

df_all = pd.concat([us, de, it, is_, tr, pl], ignore_index=True)

# Ensure numeric years/counts
df_all["year"] = pd.to_numeric(df_all["year"])
df_all["count"] = pd.to_numeric(df_all["count"])

# Panel order and titles
order = ["DE", "IT", "IS", "US", "TR", "PL"]
titles = {
    "DE": "A) Germany",
    "IT": "B) Italy",
    "IS": "C) Iceland",
    "US": "D) U.S.",
    "TR": "E) Turkey",
    "PL": "F) Poland",
}

# Shared x-axis range across all panels
xmin = int(df_all["year"].min())
xmax = int(df_all["year"].max())

# Decade ticks
start_tick = (xmin // 10) * 10
end_tick = ((xmax + 9) // 10) * 10
xticks = list(range(start_tick, end_tick + 1, 10))


plt.style.use("default")
fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
fig.patch.set_facecolor("white")

for ax, src in zip(axes, order):
    sub = df_all[df_all["source"] == src].sort_values("year")

    ax.bar(
        sub["year"],
        sub["count"],
        width=0.9,
        color="dimgray",
        edgecolor="dimgray",
        linewidth=0.3
    )

    ax.set_title(titles[src], loc="left", fontsize=16, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_xticks(xticks)

    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.grid(True, axis="x", color="0.8")
    ax.grid(False, axis="y")
    ax.tick_params(axis="both", labelsize=11)

axes[-1].set_xlabel("Year", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig('./output/textcount_trend.pdf', format='pdf', dpi=300)
#plt.savefig('./output/textcount_trend.png', format='png', dpi=300)
#plt.savefig('./output/textcount_trend.svg', format='svg', dpi=300)

plt.show()
