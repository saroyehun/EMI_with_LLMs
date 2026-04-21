# plot fig 2
import matplotlib.pyplot as plt
import seaborn as sns
#from adjustText import adjust_text
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
plt.rc("axes.spines", top=False, right=True)
matplotlib.rcParams['lines.markersize'] = 5
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams["legend.handlelength"] = 2.0
LABELSPACING = 0.6

plt.figure(figsize=(12, 9))

country_label = {'TR': 'Turkey',
                'IS': 'Iceland',
                'PL': 'Poland',
                'IT': 'Italy',
                'US': 'United States of America',
                'DE (West)': 'West Germany',
                'DE': 'Germany (from 1990)'
               }

# Prepare data
tr_df_yearly_attributed = pd.read_csv('../aggregate_data/TR/agg_sbertllms_gt1981_attributed.csv')
is_df_yearly = pd.read_csv('../aggregate_data/IS/agg_sbertllms_gt1945.csv')
pl_df_yearly = pd.read_csv('../aggregate_data/PL/agg_sbertllms_gt1989.csv')
de_df_yearly_od_lt90 = pd.read_csv('../aggregate_data/DE/agg_sbertllms_lt1990.csv')
de_df_yearly_od_ge90 = pd.read_csv('../aggregate_data/DE/agg_sbertllms_ge1990.csv')
it_df_yearly_attributed = pd.read_csv('../aggregate_data/IT/agg_sbertllms_gt1945_attributed.csv')
us_df_sessions_attributed = pd.read_csv('../aggregate_data/US/agg_sbertllms_gt1945.csv')

tr_df_yearly_attributed['source'] = 'TR'
is_df_yearly['source'] = 'IS'
pl_df_yearly['source'] = 'PL'
de_df_yearly_od_lt90['source'] = 'DE (West)'
de_df_yearly_od_ge90['source'] = 'DE'
it_df_yearly_attributed['source'] = 'IT' 
us_df_sessions_attributed['source'] = 'US'


df = pd.concat([tr_df_yearly_attributed, pl_df_yearly, de_df_yearly_od_lt90, de_df_yearly_od_ge90, it_df_yearly_attributed, is_df_yearly, us_df_sessions_attributed], 
               axis=0, ignore_index=True)
df = df[~df.ddi.isna()]



# color palette
colors = {
    'US': '#0173B2',  # Blue
    'DE': '#DE8F05',  # Orange
    'DE (West)': '#F3B562',
    'IT': '#029E73',  # Teal/Green
    'TR': '#CC78BC',  # Pink/Purple
    'PL': '#DC143C',  # Vermillion/Red-Orange
    'IS': '#8B4513',  # Bright Yellow
}

# Plot individual country data with trendlines
for country, color in colors.items():
    country_data = df[df['source'] == country]
    
    # Scatter plot
    if country == 'DE (West)':
        marker = '^'
    else:
        marker = 'o'  
    plt.scatter(country_data['evidence_intuition_diffpool_score_mean'],
                country_data['ddi'],                
               s=30, alpha=0.6, color=color, label=country_label[country], edgecolors='white',
                linewidth=0.5, marker=marker)
    r, p_val = pearsonr(country_data['evidence_intuition_diffpool_score_mean'], country_data['ddi'])
    print(f"{country}, $r = {r:.3f}$, $N = {len(country_data)}$, $p = {p_val:.3g}$")
    # Individual trendline (lighter, thinner)
    if len(country_data) > 1:
        z = np.polyfit(country_data['evidence_intuition_diffpool_score_mean'], country_data['ddi'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(country_data['evidence_intuition_diffpool_score_mean'].min(), country_data['evidence_intuition_diffpool_score_mean'].max(), 100)
        plt.plot(x_line, p(x_line), color=color, alpha=0.6, linewidth=2.5, linestyle='--')

# Overall trendline (prominent)
r, p_val = pearsonr(df['evidence_intuition_diffpool_score_mean'], df['ddi'])
print(f"Overall, $r = {r:.3f}$, $N = {len(df)}$, $p = {p_val:.3g}$")
z_overall = np.polyfit(df['evidence_intuition_diffpool_score_mean'], df['ddi'], 1)
p_overall = np.poly1d(z_overall)
x_overall = np.linspace(df['evidence_intuition_diffpool_score_mean'].min(), df['evidence_intuition_diffpool_score_mean'].max(), 100)
plt.plot(x_overall, p_overall(x_overall), color='black', linewidth=1, 
         label='Overall trend', zorder=10, alpha=0.8)

# Add confidence interval for overall trend
sns.regplot(y='ddi', x='evidence_intuition_diffpool_score_mean', data=df, 
           scatter=False, ci=95, 
           line_kws={'linewidth': 0, 'alpha':0.8},  # Hide the line (already plotted it)
           color='black', ax=plt.gca(), )

plt.xlabel("EMI")
plt.ylabel("DDI")
leg = plt.legend( bbox_to_anchor=(0.5, -0.12), title="", loc="upper center", 
           frameon=True, framealpha=0.9, ncol=3, markerscale=3)
for text, h in zip(leg.get_texts(), leg.legend_handles):
    if (text.get_text() == "Overall trend") and hasattr(h, "set_linewidth"):
        h.set_linewidth(3)
plt.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
plt.tight_layout()

plt.savefig('./output/emi_ddi_DE_unification_split.pdf', format='pdf', dpi=300)
#plt.savefig('./output/emi_ddi_unification_split.png', format='png', dpi=300)
#plt.savefig('./output/emi_ddi_unification_split.svg', format='svg', dpi=300)

plt.show()

