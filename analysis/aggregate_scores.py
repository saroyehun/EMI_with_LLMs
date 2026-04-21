# script to aggregate scores

from __future__ import annotations
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os
from scipy.stats import zscore

def linear_pooling(emi_rating, cosine_score, lam=0.5):
    return lam * emi_rating + (1 - lam) * cosine_score
    
def _bootstrap_means_streaming(
    x: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    chunk: int = 250,
) -> np.ndarray:
    
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return np.empty(0, dtype=float)

    boots = np.empty(n_boot, dtype=float)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(b, n), endpoint=False)
        boots[done : done + b] = x[idx].mean(axis=1)
        done += b

    return boots


def _bootstrap_year_worker(
    df_year: pd.DataFrame,
    year: int,
    score_cols: list[str],
    n_boot: int,
    ci: float,
    seed: int,
    chunk: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    out: list[dict] = []

    alpha = 100.0 - float(ci)
    lo_q = alpha / 2.0
    hi_q = 100.0 - (alpha / 2.0)

    for col in score_cols:
        x = df_year[col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]

        if x.size == 0:
            out.append(
                {
                    "year": int(year),
                    "score": col,
                    "mean": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                }
            )
            continue

        boots = _bootstrap_means_streaming(x, n_boot=n_boot, rng=rng, chunk=chunk)

        out.append(
            {
                "year": int(year),
                "score": col,
                "mean": float(boots.mean()),
                "ci_lower": float(np.percentile(boots, lo_q)),
                "ci_upper": float(np.percentile(boots, hi_q)),
            }
        )

    return out


def bootstrap_yearly_parallel(
    df: pd.DataFrame,
    score_cols: list[str],
    year_col: str = "year",
    n_boot: int = 10_000,
    ci: float = 95.0,
    n_jobs: int = -1,
    chunk: int = 250,
    batch_size: int = 1,
    pre_dispatch: str = "2*n_jobs",
    seed: int = 12345,
) -> pd.DataFrame:
    # Keep only needed columns to reduce payload to workers
    keep_cols = [year_col] + list(score_cols)
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_small = df[keep_cols].copy()

    years = np.array(sorted(df_small[year_col].dropna().unique()), dtype=int)
    if years.size == 0:
        return pd.DataFrame(columns=["year", "score", "mean", "ci_lower", "ci_upper"])

    # per-year seeds
    rng = np.random.default_rng(seed)
    year_seeds = rng.integers(0, 2**32 - 1, size=years.size, dtype=np.uint32)

    results_nested = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        batch_size=batch_size,
        pre_dispatch=pre_dispatch,       
        max_nbytes=None,
    )(
        delayed(_bootstrap_year_worker)(
            df_small[df_small[year_col] == int(y)],
            int(y),
            score_cols,
            n_boot,
            ci,
            int(year_seeds[i]),
            chunk,
        )
        for i, y in enumerate(years)
    )

    # Flatten
    rows = [r for year_rows in results_nested for r in year_rows]
    return pd.DataFrame(rows)
    
# VDEM data

dem_cols = ['Year', 'Clientelism Index', 'Deliberative Democracy Index',             
            'Transparent laws with predictable enforcement',
            'Judicial constraints on the executive index',            
           ]

de_dem = pd.read_csv('../aggregate_data/DE/Germany.csv', usecols=dem_cols)
it_dem = pd.read_csv('../aggregate_data/IT/Italy.csv', usecols=dem_cols)
is_dem = pd.read_csv('../aggregate_data/IS/Iceland.csv', usecols=dem_cols)
tr_dem = pd.read_csv('../aggregate_data/TR/Turkey.csv', usecols=dem_cols)

us_2025 = pd.read_csv('../aggregate_data/US/UnitedStatesofAmerica.csv',
                      usecols=dem_cols
                     )
pl_2025 = pd.read_csv('../aggregate_data/PL/Poland.csv',
                      usecols=dem_cols
                     )
                     
#histrical GDP data
#project maddison
df_gdp_per_capita_till2022 = pd.read_csv("https://ourworldindata.org/grapher/gdp-per-capita-maddison-project-database.csv?v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
df_gdp_per_capita_till2022.rename(columns = {'year': 'Year'}, inplace=True)
df_gdp_per_capita_till2022 = df_gdp_per_capita_till2022[['entity', 'Year', 'gdp_per_capita']]

us_dem = us_2025.join(df_gdp_per_capita_till2022[df_gdp_per_capita_till2022.entity == 'United States' ].set_index('Year'), on='Year', how='left')
de_dem = de_dem.join(df_gdp_per_capita_till2022[df_gdp_per_capita_till2022.entity == 'Germany' ].set_index('Year'), on='Year', how='left')
it_dem = it_dem.join(df_gdp_per_capita_till2022[df_gdp_per_capita_till2022.entity == 'Italy' ].set_index('Year'), on='Year', how='left')
is_dem = is_dem.join(df_gdp_per_capita_till2022[df_gdp_per_capita_till2022.entity == 'Iceland'].set_index('Year'), on='Year', how='left')
pl_dem = pl_2025.join(df_gdp_per_capita_till2022[df_gdp_per_capita_till2022.entity == 'Poland'].set_index('Year'), on='Year', how='left')
tr_dem = tr_dem.join(df_gdp_per_capita_till2022[df_gdp_per_capita_till2022.entity == 'Turkey'].set_index('Year'), on='Year', how='left')

# US
cols = ['row_index', 'speech_id', 'year', 'procedural', 'text', 'avg_evidence_score','avg_intuition_score', 'chamber_harmonized']
us_sbert = pd.read_csv('../data/US/US_congress_gt1945_attributed_bioguide_party_procedural_chunk_sbertmgte.csv',
                       usecols=cols
                      )

# 3 llms
cols = ['row_index', 'speech_id', 'evidence_based','evidence_free', 'year']
us_llama = pd.read_csv('../data/US/Meta-Llama-3.1-8B-Instruct_results.csv',
                       usecols=cols,
                      )
us_qwen = pd.read_csv('../data/US/Qwen2.5-7B-Instruct_results.csv',
                       usecols=cols,
                      )
us_apertus = pd.read_csv('../data/US/Apertus-8B-Instruct-2509_results.csv',
                       usecols=cols,
                      )
                      

us_merged = (
    us_llama.set_index("row_index")
    .join(us_qwen.set_index("row_index")[["evidence_based", "evidence_free"]], rsuffix="_qwen")
    .join(us_apertus.set_index("row_index")[["evidence_based", "evidence_free"]], rsuffix="_apertus")
)

# Compute averages
us_merged["evidence_avg"] = us_merged[["evidence_based", "evidence_based_qwen", "evidence_based_apertus"]].mean(axis=1, skipna=True)
us_merged["intuition_avg"] = us_merged[["evidence_free", "evidence_free_qwen", "evidence_free_apertus"]].mean(axis=1, skipna=True)
us_merged = us_merged.reset_index()
us_merged["evidence_avg"] = us_merged["evidence_avg"] / 4
us_merged["intuition_avg"] = us_merged["intuition_avg"] / 4
us_merged["EMI_avg_diff_llms"] = us_merged["evidence_avg"] - us_merged["intuition_avg"]

us_sbert['evidence_emb_z'] = zscore(us_sbert['avg_evidence_score'], nan_policy='omit')
us_sbert['intuition_emb_z'] = zscore(us_sbert['avg_intuition_score'], nan_policy='omit')
us_sbert['emi_embdiff'] = us_sbert['avg_evidence_score'] - us_sbert['avg_intuition_score']
us_sbert['emi_embdiff_z'] = zscore(us_sbert['emi_embdiff'], nan_policy='omit')
us_merged["EMI_avg_diff_llms_z"] = zscore(us_merged['EMI_avg_diff_llms'], nan_policy='omit')
us_merged["evidence_avg_z"] = zscore(us_merged['evidence_avg'], nan_policy='omit')
us_merged["intuition_avg_z"] = zscore(us_merged['intuition_avg'], nan_policy='omit')

us_sbert_3llms = us_sbert[us_sbert.chamber_harmonized != 'E'].join(us_merged[['EMI_avg_diff_llms_z', 'evidence_avg_z', 'intuition_avg_z', 'row_index']].set_index('row_index'),
                                  on='row_index', how='left'
                                 )
                                 
#linear pooling
us_sbert_3llms['rating_emb_pool'] = linear_pooling(us_sbert_3llms['emi_embdiff_z'], us_sbert_3llms['EMI_avg_diff_llms_z'])
us_sbert_3llms['rating_emb_pool_z'] = zscore(us_sbert_3llms['rating_emb_pool'], nan_policy='omit')
us_sbert_3llms['ev_pool'] = linear_pooling(us_sbert_3llms['evidence_avg_z'], us_sbert_3llms['evidence_emb_z'])
us_sbert_3llms['ev_pool_z'] = zscore(us_sbert_3llms['ev_pool'], nan_policy='omit')

us_sbert_3llms['int_pool'] = linear_pooling(us_sbert_3llms['intuition_avg_z'], us_sbert_3llms['intuition_emb_z'])
us_sbert_3llms['int_pool_z'] = zscore(us_sbert_3llms['int_pool'], nan_policy='omit')


us_df_yearly = (us_sbert_3llms.groupby('year')
                             .agg(                                 
                                 evidence_intuition_diffpool_score_mean = ('rating_emb_pool_z', 'mean'),
                                 evidence_mean                      = ('ev_pool_z', 'mean'),
                                 intuition_mean                     = ('int_pool_z', 'mean'),
                                 count                                  = ('int_pool_z', 'count'),
                                 )
                             .reset_index()
                            )

us_ddi = us_dem[['Year', 'Deliberative Democracy Index',                
                 'Transparent laws with predictable enforcement', 'Clientelism Index', 'gdp_per_capita', 
                'Judicial constraints on the executive index',                 
               ]]

us_ddi.rename(columns = {'Year':'year', 'Deliberative Democracy Index':'ddi',                         
                         'Transparent laws with predictable enforcement':'transparent_laws',
                         'Clientelism Index':'clientelism_index', 
                         'Judicial constraints on the executive index':'judiciary_index',                         
                        }, inplace=True
             )

us_df_yearly = us_df_yearly.join(us_ddi.set_index('year'), on='year', how='left')
us_df_yearly['country'] = 'United States'
us_df_yearly.to_csv('../aggregate_data/US/agg_sbertllms_gt1945.csv', index=False)
#bootsrap
score_cols = ['ev_pool_z', 'rating_emb_pool_z', 'int_pool_z']
boot_df = bootstrap_yearly_parallel(
        df=us_sbert_3llms,
        score_cols=score_cols,
        year_col="year",
        n_boot=10_000,
        ci=95,
        n_jobs=50,
        chunk=250,         
        batch_size=1,
        pre_dispatch="2*n_jobs",
        seed=12345,
    )


# boot df for plotting, need to join with ddi
boot_df = boot_df.join(us_ddi.set_index('year')[['ddi']], on='year', how='left')
boot_df.to_csv('../aggregate_data/US/bootstrapyearly_sbertllms_gt1945.csv', index=False)

# DE
# sbert
cols = ['speech_id','seq_id','year','text','procedural','avg_evidence_score','avg_intuition_score', 'evidence_minus_intuition_score']
de_sbert_od = pd.read_csv('../data/DE/opendiscourse_gt1945_top100proceduralfiltered_emi_sbertmgte.csv',
                       usecols=cols
                      )
cols = ['row_index', 'seq_id', 'evidence_based','evidence_free', 'year']
de_llama_od = pd.read_csv('../data/DE/Meta-Llama-3.1-8B-Instruct_results.csv',
                       usecols=cols,
                      )
de_qwen_od = pd.read_csv('../data/DE/Qwen2.5-7B-Instruct_results.csv',
                       usecols=cols,
                      )
de_apertus_od = pd.read_csv('../data/DE/Apertus-8B-Instruct-2509_results.csv',
                       usecols=cols,
                      )
de_merged_od = (
    de_llama_od.set_index("seq_id")
    .join(de_qwen_od.set_index("seq_id")[["evidence_based", "evidence_free"]], rsuffix="_qwen")
    .join(de_apertus_od.set_index("seq_id")[["evidence_based", "evidence_free"]], rsuffix="_apertus")
)
# subset de_sbert_od with year
de_sbert_od_lt90 = de_sbert_od[de_sbert_od.year < 1990]
# subset de_merge_od with year
de_merged_od_lt90 = de_merged_od[de_merged_od.year < 1990]
# then do norm and combine scores
de_merged_od_lt90["evidence_avg"] = de_merged_od_lt90[["evidence_based", "evidence_based_qwen", "evidence_based_apertus"]].mean(axis=1, skipna=True)
de_merged_od_lt90["intuition_avg"] = de_merged_od_lt90[["evidence_free", "evidence_free_qwen", "evidence_free_apertus"]].mean(axis=1, skipna=True)
de_merged_od_lt90 = de_merged_od_lt90.reset_index()
de_merged_od_lt90["evidence_avg"] = de_merged_od_lt90["evidence_avg"] / 4
de_merged_od_lt90["intuition_avg"] = de_merged_od_lt90["intuition_avg"] / 4
de_merged_od_lt90["EMI_avg_diff_llms"] = de_merged_od_lt90["evidence_avg"] - de_merged_od_lt90["intuition_avg"]

de_sbert_od_lt90['evidence_emb_z'] = zscore(de_sbert_od_lt90['avg_evidence_score'], nan_policy='omit')
de_sbert_od_lt90['intuition_emb_z'] = zscore(de_sbert_od_lt90['avg_intuition_score'], nan_policy='omit')
de_sbert_od_lt90['emi_embdiff'] = de_sbert_od_lt90['avg_evidence_score'] - de_sbert_od_lt90['avg_intuition_score']
de_sbert_od_lt90['emi_embdiff_z'] = zscore(de_sbert_od_lt90['emi_embdiff'], nan_policy='omit')
de_merged_od_lt90["EMI_avg_diff_llms_z"] = zscore(de_merged_od_lt90['EMI_avg_diff_llms'], nan_policy='omit')
de_merged_od_lt90["evidence_avg_z"] = zscore(de_merged_od_lt90['evidence_avg'], nan_policy='omit')
de_merged_od_lt90["intuition_avg_z"] = zscore(de_merged_od_lt90['intuition_avg'], nan_policy='omit')

de_sbert_3llms_od_lt90 = de_sbert_od_lt90.join(de_merged_od_lt90[['EMI_avg_diff_llms_z', 'evidence_avg_z', 'intuition_avg_z', 'seq_id']].set_index('seq_id'),
                                  on='seq_id', how='left'
                                 )

#linear pooling
de_sbert_3llms_od_lt90['rating_emb_pool'] = linear_pooling(de_sbert_3llms_od_lt90['emi_embdiff_z'], de_sbert_3llms_od_lt90['EMI_avg_diff_llms_z'])
de_sbert_3llms_od_lt90['rating_emb_pool_z'] = zscore(de_sbert_3llms_od_lt90['rating_emb_pool'], nan_policy='omit')
de_sbert_3llms_od_lt90['ev_pool'] = linear_pooling(de_sbert_3llms_od_lt90['evidence_avg_z'], de_sbert_3llms_od_lt90['evidence_emb_z'])
de_sbert_3llms_od_lt90['ev_pool_z'] = zscore(de_sbert_3llms_od_lt90['ev_pool'], nan_policy='omit')

de_sbert_3llms_od_lt90['int_pool'] = linear_pooling(de_sbert_3llms_od_lt90['intuition_avg_z'], de_sbert_3llms_od_lt90['intuition_emb_z'])
de_sbert_3llms_od_lt90['int_pool_z'] = zscore(de_sbert_3llms_od_lt90['int_pool'], nan_policy='omit')
de_sbert_3llms_od_lt90['pooled_diff'] = de_sbert_3llms_od_lt90['ev_pool_z'] - de_sbert_3llms_od_lt90['int_pool_z']
de_sbert_3llms_od_lt90['pooled_diff_z'] = zscore(de_sbert_3llms_od_lt90['pooled_diff'], nan_policy='omit')

de_df_yearly_od_lt90 = (de_sbert_3llms_od_lt90.groupby('year')
                             .agg(
                                 evidence_intuition_diffpool_score_mean = ('rating_emb_pool_z', 'mean'),                                 
                                 evidence_mean                      = ('ev_pool_z', 'mean'),
                                 intuition_mean                     = ('int_pool_z', 'mean'),
                                 count                                  = ('int_pool_z', 'count'),
                                 )
                             .reset_index()
                            )

de_ddi = de_dem[['Year', 'Deliberative Democracy Index',
                 'Transparent laws with predictable enforcement',
                 'Clientelism Index', 'gdp_per_capita', 
                 'Judicial constraints on the executive index',                                 
               ]]
de_ddi.rename(columns = {'Year':'year', 'Deliberative Democracy Index':'ddi',                                         
                         'Transparent laws with predictable enforcement':'transparent_laws',
                         'Clientelism Index':'clientelism_index', 
                         'Judicial constraints on the executive index':'judiciary_index',                         
                        }, inplace=True
             )

de_df_yearly_od_lt90 = de_df_yearly_od_lt90.join(de_ddi.set_index('year'), on='year', how='left')
de_df_yearly_od_lt90['country'] = 'West Germany'
de_df_yearly_od_lt90.to_csv('../aggregate_data/DE/agg_sbertllms_lt1990.csv', index=False)

score_cols = ['ev_pool_z', 'rating_emb_pool_z', 'int_pool_z']
boot_df_de_od_lt90 = bootstrap_yearly_parallel(
        df=de_sbert_3llms_od_lt90,
        score_cols=score_cols,
        year_col="year",
        n_boot=10_000,
        ci=95,
        n_jobs=50,
        chunk=250,         
        batch_size=1,
        pre_dispatch="2*n_jobs",
        seed=12345,
    )
# join bootdf with ddi and save to file
boot_df_de_od_lt90 = boot_df_de_od_lt90.join(de_ddi.set_index('year')[['ddi']], on='year', how='left')
boot_df_de_od_lt90.to_csv('../aggregate_data/DE/bootstrapyearly_sbertllms_lt1990.csv', index=False)

# split ge 1990
de_sbert_od_ge90 = de_sbert_od[de_sbert_od.year >= 1990]
# subset de_merge_od with year
de_merged_od_ge90 = de_merged_od[de_merged_od.year >= 1990]

de_merged_od_ge90["evidence_avg"] = de_merged_od_ge90[["evidence_based", "evidence_based_qwen", "evidence_based_apertus"]].mean(axis=1, skipna=True)
de_merged_od_ge90["intuition_avg"] = de_merged_od_ge90[["evidence_free", "evidence_free_qwen", "evidence_free_apertus"]].mean(axis=1, skipna=True)
de_merged_od_ge90 = de_merged_od_ge90.reset_index()
de_merged_od_ge90["evidence_avg"] = de_merged_od_ge90["evidence_avg"] / 4
de_merged_od_ge90["intuition_avg"] = de_merged_od_ge90["intuition_avg"] / 4
de_merged_od_ge90["EMI_avg_diff_llms"] = de_merged_od_ge90["evidence_avg"] - de_merged_od_ge90["intuition_avg"]

de_sbert_od_ge90['evidence_emb_z'] = zscore(de_sbert_od_ge90['avg_evidence_score'], nan_policy='omit')
de_sbert_od_ge90['intuition_emb_z'] = zscore(de_sbert_od_ge90['avg_intuition_score'], nan_policy='omit')
de_sbert_od_ge90['emi_embdiff'] = de_sbert_od_ge90['avg_evidence_score'] - de_sbert_od_ge90['avg_intuition_score']
de_sbert_od_ge90['emi_embdiff_z'] = zscore(de_sbert_od_ge90['emi_embdiff'], nan_policy='omit')
de_merged_od_ge90["EMI_avg_diff_llms_z"] = zscore(de_merged_od_ge90['EMI_avg_diff_llms'], nan_policy='omit')
de_merged_od_ge90["evidence_avg_z"] = zscore(de_merged_od_ge90['evidence_avg'], nan_policy='omit')
de_merged_od_ge90["intuition_avg_z"] = zscore(de_merged_od_ge90['intuition_avg'], nan_policy='omit')

de_sbert_3llms_od_ge90 = de_sbert_od_ge90.join(de_merged_od_ge90[['EMI_avg_diff_llms_z', 'evidence_avg_z', 'intuition_avg_z', 'seq_id']].set_index('seq_id'),
                                  on='seq_id', how='left'
                                 )

#linear pooling
de_sbert_3llms_od_ge90['rating_emb_pool'] = linear_pooling(de_sbert_3llms_od_ge90['emi_embdiff_z'], de_sbert_3llms_od_ge90['EMI_avg_diff_llms_z'])
de_sbert_3llms_od_ge90['rating_emb_pool_z'] = zscore(de_sbert_3llms_od_ge90['rating_emb_pool'], nan_policy='omit')
de_sbert_3llms_od_ge90['ev_pool'] = linear_pooling(de_sbert_3llms_od_ge90['evidence_avg_z'], de_sbert_3llms_od_ge90['evidence_emb_z'])
de_sbert_3llms_od_ge90['ev_pool_z'] = zscore(de_sbert_3llms_od_ge90['ev_pool'], nan_policy='omit')

de_sbert_3llms_od_ge90['int_pool'] = linear_pooling(de_sbert_3llms_od_ge90['intuition_avg_z'], de_sbert_3llms_od_ge90['intuition_emb_z'])
de_sbert_3llms_od_ge90['int_pool_z'] = zscore(de_sbert_3llms_od_ge90['int_pool'], nan_policy='omit')
de_sbert_3llms_od_ge90['pooled_diff'] = de_sbert_3llms_od_ge90['ev_pool_z'] - de_sbert_3llms_od_ge90['int_pool_z']
de_sbert_3llms_od_ge90['pooled_diff_z'] = zscore(de_sbert_3llms_od_ge90['pooled_diff'], nan_policy='omit')

de_df_yearly_od_ge90 = (de_sbert_3llms_od_ge90.groupby('year')
                             .agg(
                                 evidence_intuition_diffpool_score_mean = ('rating_emb_pool_z', 'mean'),                                 
                                 evidence_mean                      = ('ev_pool_z', 'mean'),
                                 intuition_mean                     = ('int_pool_z', 'mean'),
                                 count                                  = ('int_pool_z', 'count'),
                                 )
                             .reset_index()
                            )

de_ddi = de_dem[['Year', 'Deliberative Democracy Index',                 
                 'Transparent laws with predictable enforcement', 
                 'Clientelism Index', 'gdp_per_capita', 
                 'Judicial constraints on the executive index',                 
               ]]
de_ddi.rename(columns = {'Year':'year', 'Deliberative Democracy Index':'ddi',                         
                         'Transparent laws with predictable enforcement':'transparent_laws',
                         'Clientelism Index':'clientelism_index', 
                         'Judicial constraints on the executive index':'judiciary_index',                         
                        }, inplace=True
             )

de_df_yearly_od_ge90 = de_df_yearly_od_ge90.join(de_ddi.set_index('year'), on='year', how='left')

de_df_yearly_od_ge90['country'] = 'Germany'
de_df_yearly_od_ge90.to_csv('../aggregate_data/DE/agg_sbertllms_ge1990.csv', index=False)

boot_df_de_od_ge90 = bootstrap_yearly_parallel(
        df=de_sbert_3llms_od_ge90,
        score_cols=score_cols,
        year_col="year",
        n_boot=10_000,
        ci=95,
        n_jobs=50,
        chunk=250,          
        batch_size=1,
        pre_dispatch="2*n_jobs",
        seed=12345,
    )
    
# save the boot df joined with ddi to file for  plotting
# join bootdf with ddi and save to file
boot_df_de_od_ge90 = boot_df_de_od_ge90.join(de_ddi.set_index('year')[['ddi']], on='year', how='left')
boot_df_de_od_ge90.to_csv('../aggregate_data/DE/bootstrapyearly_sbertllms_ge1990.csv', index=False)

# IT attributed

#sbert
cols = ['date','chunk_uid','year','procedural','avg_evidence_score','avg_intuition_score', 'text']
it_sbert_attributed = pd.read_csv('../data/IT/attributed_chunks_gt1945_sbertmgte.csv',
                       usecols=cols
                      )
                      
# 3 llms
cols = ['row_index', 'chunk_uid', 'evidence_based','evidence_free', 'year']
it_llama_attributed = pd.read_csv('../data/IT/Meta-Llama-3.1-8B-Instruct_results.csv',
                       usecols=cols,
                      )
it_qwen_attributed = pd.read_csv('../data/IT/Qwen2.5-7B-Instruct_results.csv',
                       usecols=cols,
                      )
it_apertus_attributed = pd.read_csv('../data/IT/Apertus-8B-Instruct-2509_results.csv',
                       usecols=cols,
                      )

it_merged_attributed = (
    it_llama_attributed.set_index("row_index")
    .join(it_qwen_attributed.set_index("row_index")[["evidence_based", "evidence_free"]], rsuffix="_qwen")
    .join(it_apertus_attributed.set_index("row_index")[["evidence_based", "evidence_free"]], rsuffix="_apertus")
)

# Compute averages
it_merged_attributed["evidence_avg"] = it_merged_attributed[["evidence_based", "evidence_based_qwen", "evidence_based_apertus"]].mean(axis=1, skipna=True)
it_merged_attributed["intuition_avg"] = it_merged_attributed[["evidence_free", "evidence_free_qwen", "evidence_free_apertus"]].mean(axis=1, skipna=True)
it_merged_attributed = it_merged_attributed.reset_index()
it_merged_attributed["evidence_avg"] = it_merged_attributed["evidence_avg"] / 4
it_merged_attributed["intuition_avg"] = it_merged_attributed["intuition_avg"] / 4
it_merged_attributed["EMI_avg_diff_llms"] = it_merged_attributed["evidence_avg"] - it_merged_attributed["intuition_avg"]

it_sbert_attributed['evidence_emb_z'] = zscore(it_sbert_attributed['avg_evidence_score'], nan_policy='omit')
it_sbert_attributed['intuition_emb_z'] = zscore(it_sbert_attributed['avg_intuition_score'], nan_policy='omit')
it_sbert_attributed['emi_embdiff'] = it_sbert_attributed['avg_evidence_score'] - it_sbert_attributed['avg_intuition_score']
it_sbert_attributed['emi_embdiff_z'] = zscore(it_sbert_attributed['emi_embdiff'], nan_policy='omit')
it_merged_attributed["EMI_avg_diff_llms_z"] = zscore(it_merged_attributed['EMI_avg_diff_llms'], nan_policy='omit')
it_merged_attributed["evidence_avg_z"] = zscore(it_merged_attributed['evidence_avg'], nan_policy='omit')
it_merged_attributed["intuition_avg_z"] = zscore(it_merged_attributed['intuition_avg'], nan_policy='omit')


it_sbert_3llms_attributed = it_sbert_attributed.join(it_merged_attributed[['EMI_avg_diff_llms_z', 'evidence_avg_z', 'intuition_avg_z', 'chunk_uid']].set_index('chunk_uid'),
                                  on='chunk_uid', how='left'
                                 )
                                 
#linear pooling
it_sbert_3llms_attributed['rating_emb_pool'] = linear_pooling(it_sbert_3llms_attributed['emi_embdiff_z'], it_sbert_3llms_attributed['EMI_avg_diff_llms_z'])
it_sbert_3llms_attributed['rating_emb_pool_z'] = zscore(it_sbert_3llms_attributed['rating_emb_pool'], nan_policy='omit')
it_sbert_3llms_attributed['ev_pool'] = linear_pooling(it_sbert_3llms_attributed['evidence_avg_z'], it_sbert_3llms_attributed['evidence_emb_z'])
it_sbert_3llms_attributed['ev_pool_z'] = zscore(it_sbert_3llms_attributed['ev_pool'], nan_policy='omit')

it_sbert_3llms_attributed['int_pool'] = linear_pooling(it_sbert_3llms_attributed['intuition_avg_z'], it_sbert_3llms_attributed['intuition_emb_z'])
it_sbert_3llms_attributed['int_pool_z'] = zscore(it_sbert_3llms_attributed['int_pool'], nan_policy='omit')

boot_df_it_attributed = bootstrap_yearly_parallel(
        df=it_sbert_3llms_attributed,
        score_cols=score_cols,
        year_col="year",
        n_boot=10_000,
        ci=95,
        n_jobs=50,
        chunk=250,
        batch_size=1,
        pre_dispatch="2*n_jobs",
        seed=12345,
    )


it_df_yearly_attributed = (it_sbert_3llms_attributed.groupby('year')
                             .agg(
                                 evidence_intuition_diffpool_score_mean = ('rating_emb_pool_z', 'mean'),
                                 evidence_mean                      = ('ev_pool_z', 'mean'),
                                 intuition_mean                     = ('int_pool_z', 'mean'),
                                 count                                  = ('int_pool_z', 'count'),
                                 )
                             .reset_index()
                            )

it_ddi = it_dem[['Year', 'Deliberative Democracy Index',                 
                 'Transparent laws with predictable enforcement', 
                 'Clientelism Index', 'gdp_per_capita', 
                 'Judicial constraints on the executive index',                                
               ]]
it_ddi.rename(columns = {'Year':'year', 'Deliberative Democracy Index':'ddi', 
                         'Transparent laws with predictable enforcement':'transparent_laws',
                         'Clientelism Index':'clientelism_index', 
                         'Judicial constraints on the executive index':'judiciary_index',                         
                        }, inplace=True
             )

it_df_yearly_attributed = it_df_yearly_attributed.join(it_ddi.set_index('year'), on='year', how='left')

it_df_yearly_attributed['country'] = 'Italy'
it_df_yearly_attributed.to_csv('../aggregate_data/IT/agg_sbertllms_gt1945_attributed.csv', index=False)

# save boot joined with ddi
boot_df_it_attributed = boot_df_it_attributed.join(it_ddi.set_index('year')[['ddi']], on='year', how='left')
boot_df_it_attributed.to_csv('../aggregate_data/IT/bootstrapyearly_sbertllms_gt1945.csv', index=False)

# IS
#sbert
cols = ['chunk_index', 'procedural', 'year','avg_evidence_score','avg_intuition_score', 'text']
is_sbert = pd.read_csv('../data/IS/IGC_chunks_top100proceduralfiltered_gt1945_sbertmgte.csv',
                       usecols=cols
                      )
# 3 llms
cols = ['row_index', 'chunk_index', 'evidence_based','evidence_free', 'year', 'procedural']
is_llama = pd.read_csv('../data/IS/Meta-Llama-3.1-8B-Instruct_results.csv',
                       usecols=cols,
                      )
is_qwen = pd.read_csv('../data/IS/Qwen2.5-7B-Instruct_results.csv',
                       usecols=cols,
                      )
is_apertus = pd.read_csv('../data/IS/Apertus-8B-Instruct-2509_results.csv',
                       usecols=cols,
                      )
                      
is_merged = (
    is_llama.set_index("row_index")
    .join(is_qwen.set_index("row_index")[["evidence_based", "evidence_free"]], rsuffix="_qwen")
    .join(is_apertus.set_index("row_index")[["evidence_based", "evidence_free"]], rsuffix="_apertus")
)

# Compute averages
is_merged["evidence_avg"] = is_merged[["evidence_based", "evidence_based_qwen", "evidence_based_apertus"]].mean(axis=1, skipna=True)
is_merged["intuition_avg"] = is_merged[["evidence_free", "evidence_free_qwen", "evidence_free_apertus"]].mean(axis=1, skipna=True)
is_merged = is_merged.reset_index()
is_merged["evidence_avg"] = is_merged["evidence_avg"] / 4
is_merged["intuition_avg"] = is_merged["intuition_avg"] / 4
is_merged["EMI_avg_diff_llms"] = is_merged["evidence_avg"] - is_merged["intuition_avg"]

is_sbert['evidence_emb_z'] = zscore(is_sbert['avg_evidence_score'], nan_policy='omit')
is_sbert['intuition_emb_z'] = zscore(is_sbert['avg_intuition_score'], nan_policy='omit')
is_sbert['emi_embdiff'] = is_sbert['avg_evidence_score'] - is_sbert['avg_intuition_score']
is_sbert['emi_embdiff_z'] = zscore(is_sbert['emi_embdiff'], nan_policy='omit')
is_merged["EMI_avg_diff_llms_z"] = zscore(is_merged['EMI_avg_diff_llms'], nan_policy='omit')
is_merged["evidence_avg_z"] = zscore(is_merged['evidence_avg'], nan_policy='omit')
is_merged["intuition_avg_z"] = zscore(is_merged['intuition_avg'], nan_policy='omit')

is_sbert_3llms = is_sbert.join(is_merged[['EMI_avg_diff_llms_z', 'evidence_avg_z', 'intuition_avg_z', 'chunk_index']].set_index(['chunk_index']),
                                  on=['chunk_index',], how='left'
                                 )

#linear pooling
is_sbert_3llms['rating_emb_pool'] = linear_pooling(is_sbert_3llms['emi_embdiff_z'], is_sbert_3llms['EMI_avg_diff_llms_z'])
is_sbert_3llms['rating_emb_pool_z'] = zscore(is_sbert_3llms['rating_emb_pool'], nan_policy='omit')
is_sbert_3llms['ev_pool'] = linear_pooling(is_sbert_3llms['evidence_avg_z'], is_sbert_3llms['evidence_emb_z'])
is_sbert_3llms['ev_pool_z'] = zscore(is_sbert_3llms['ev_pool'], nan_policy='omit')

is_sbert_3llms['int_pool'] = linear_pooling(is_sbert_3llms['intuition_avg_z'], is_sbert_3llms['intuition_emb_z'])
is_sbert_3llms['int_pool_z'] = zscore(is_sbert_3llms['int_pool'], nan_policy='omit')

boot_df_is = bootstrap_yearly_parallel(
        df=is_sbert_3llms,
        score_cols=score_cols,
        year_col="year",
        n_boot=10_000,
        ci=95,
        n_jobs=50,
        chunk=250,          
        batch_size=1,
        pre_dispatch="2*n_jobs",
        seed=12345,
    )

is_df_yearly = (is_sbert_3llms.groupby('year')
                             .agg(                                 
                                 evidence_intuition_diffpool_score_mean = ('rating_emb_pool_z', 'mean'),
                                 evidence_mean                      = ('ev_pool_z', 'mean'),
                                 intuition_mean                     = ('int_pool_z', 'mean'),
                                 count                                  = ('int_pool_z', 'count'),
                                 )
                             .reset_index()
                            )

is_ddi = is_dem[['Year', 'Deliberative Democracy Index',
                 'Transparent laws with predictable enforcement', 
                 'Clientelism Index', 'gdp_per_capita', 
                 'Judicial constraints on the executive index',                                 
               ]]
is_ddi.rename(columns = {'Year':'year', 'Deliberative Democracy Index':'ddi', 
                         'Transparent laws with predictable enforcement':'transparent_laws',
                         'Clientelism Index':'clientelism_index', 
                         'Judicial constraints on the executive index':'judiciary_index',                         
                        }, inplace=True
             )

is_df_yearly = is_df_yearly.join(is_ddi.set_index('year'), on='year', how='left')
is_df_yearly['country'] = 'Iceland'
is_df_yearly.to_csv('../aggregate_data/IS/agg_sbertllms_gt1945.csv', index=False)

# join with ddi and save to file
boot_df_is = boot_df_is.join(is_ddi.set_index('year')[['ddi']], on='year', how='left')
boot_df_is.to_csv('../data/IS/bootstrapyearly_sbertllms_gt1945.csv', index=False)

# PL

#sbert
cols = ['chunk_index','text','date','year','avg_evidence_score','avg_intuition_score', 'procedural']
pl_sbert = pd.read_csv('../data/PL/tei_gt1989_sbertmgte.csv',
                       usecols=cols
                      )

# 3 llms
cols = ['row_index', 'chunk_index', 'evidence_based','evidence_free', 'year']
pl_llama = pd.read_csv('../data/PL/Meta-Llama-3.1-8B-Instruct_results.csv',
                       usecols=cols,
                      )
pl_qwen = pd.read_csv('../data/PL/Qwen2.5-7B-Instruct_results.csv',
                       usecols=cols,
                      )
pl_apertus = pd.read_csv('../data/PL/Apertus-8B-Instruct-2509_results.csv',
                       usecols=cols,
                      )

pl_merged = (
    pl_llama.set_index("row_index")
    .join(pl_qwen.set_index("row_index")[["evidence_based", "evidence_free"]], rsuffix="_qwen")
    .join(pl_apertus.set_index("row_index")[["evidence_based", "evidence_free"]], rsuffix="_apertus")
)

# Compute averages
pl_merged["evidence_avg"] = pl_merged[["evidence_based", "evidence_based_qwen", "evidence_based_apertus"]].mean(axis=1, skipna=True)
pl_merged["intuition_avg"] = pl_merged[["evidence_free", "evidence_free_qwen", "evidence_free_apertus"]].mean(axis=1, skipna=True)
pl_merged = pl_merged.reset_index()
pl_merged["evidence_avg"] = pl_merged["evidence_avg"] / 4
pl_merged["intuition_avg"] = pl_merged["intuition_avg"] / 4
pl_merged["EMI_avg_diff_llms"] = pl_merged["evidence_avg"] - pl_merged["intuition_avg"]

pl_sbert['evidence_emb_z'] = zscore(pl_sbert['avg_evidence_score'], nan_policy='omit')
pl_sbert['intuition_emb_z'] = zscore(pl_sbert['avg_intuition_score'], nan_policy='omit')
pl_sbert['emi_embdiff'] = pl_sbert['avg_evidence_score'] - pl_sbert['avg_intuition_score']
pl_sbert['emi_embdiff_z'] = zscore(pl_sbert['emi_embdiff'], nan_policy='omit')
pl_merged["EMI_avg_diff_llms_z"] = zscore(pl_merged['EMI_avg_diff_llms'], nan_policy='omit')
pl_merged["evidence_avg_z"] = zscore(pl_merged['evidence_avg'], nan_policy='omit')
pl_merged["intuition_avg_z"] = zscore(pl_merged['intuition_avg'], nan_policy='omit')

pl_sbert_3llms = pl_sbert.join(pl_merged[['EMI_avg_diff_llms_z', 'evidence_avg_z', 'intuition_avg_z', 'chunk_index']].set_index(['chunk_index']),
                                  on=['chunk_index'], how='left'
                                 )

#linear pooling
pl_sbert_3llms['rating_emb_pool'] = linear_pooling(pl_sbert_3llms['emi_embdiff_z'], pl_sbert_3llms['EMI_avg_diff_llms_z'])
pl_sbert_3llms['rating_emb_pool_z'] = zscore(pl_sbert_3llms['rating_emb_pool'], nan_policy='omit')
pl_sbert_3llms['ev_pool'] = linear_pooling(pl_sbert_3llms['evidence_avg_z'], pl_sbert_3llms['evidence_emb_z'])
pl_sbert_3llms['ev_pool_z'] = zscore(pl_sbert_3llms['ev_pool'], nan_policy='omit')

pl_sbert_3llms['int_pool'] = linear_pooling(pl_sbert_3llms['intuition_avg_z'], pl_sbert_3llms['intuition_emb_z'])
pl_sbert_3llms['int_pool_z'] = zscore(pl_sbert_3llms['int_pool'], nan_policy='omit')

boot_df_pl = bootstrap_yearly_parallel(
        df=pl_sbert_3llms,
        score_cols=score_cols,
        year_col="year",
        n_boot=10_000,
        ci=95,
        n_jobs=50,
        chunk=250,          
        batch_size=1,
        pre_dispatch="2*n_jobs",
        seed=12345,
    )
pl_df_yearly = (pl_sbert_3llms.groupby('year')
                             .agg(                                 
                                 evidence_intuition_diffpool_score_mean = ('rating_emb_pool_z', 'mean'),
                                 evidence_mean                      = ('ev_pool_z', 'mean'),
                                 intuition_mean                     = ('int_pool_z', 'mean'),
                                 count                                  = ('int_pool_z', 'count'),
                                 )
                             .reset_index()
                            )

pl_ddi = pl_dem[['Year', 'Deliberative Democracy Index',
                 'Transparent laws with predictable enforcement', 'Clientelism Index', 'gdp_per_capita', 
                 'Judicial constraints on the executive index',             
               ]]
pl_ddi.rename(columns = {'Year':'year', 'Deliberative Democracy Index':'ddi', 
                         'Transparent laws with predictable enforcement':'transparent_laws',
                         'Clientelism Index':'clientelism_index', 
                         'Judicial constraints on the executive index':'judiciary_index',                        
                        }, inplace=True
             )

pl_df_yearly = pl_df_yearly.join(pl_ddi.set_index('year'), on='year', how='left')

pl_df_yearly['country'] = 'Poland'
pl_df_yearly.to_csv('../aggregate_data/PL/agg_sbertllms_gt1989.csv', index=False)

# bootdf join with ddi and save
boot_df_pl = boot_df_pl.join(pl_ddi.set_index('year')[['ddi']], on='year', how='left')
boot_df_pl.to_csv('../aggregate_data/PL/bootstrapyearly_sbertllms_gt1989.csv', index=False)

# TR
#sbert
cols = ['chunk_uid','text','year','avg_evidence_score','avg_intuition_score', 'procedural']
tr_sbert_attributed = pd.read_csv('../data/TR/attributed_tbmm_gt1981_filteredtop100_procedural_filtered_sbertmgte.csv',
                       usecols=cols
                      )

# 3 llms
cols = ['row_index', 'chunk_uid', 'evidence_based','evidence_free', 'year']
tr_llama_attributed = pd.read_csv('../data/TR/Meta-Llama-3.1-8B-Instruct_results.csv',
                       usecols=cols,
                      )
tr_qwen_attributed = pd.read_csv('../data/TR/Qwen2.5-7B-Instruct_results.csv',
                       usecols=cols,
                      )
tr_apertus_attributed = pd.read_csv('../data/TR/Apertus-8B-Instruct-2509_results.csv',
                       usecols=cols,
                      )

tr_merged_attributed = (
    tr_llama_attributed.set_index("row_index")
    .join(tr_qwen_attributed.set_index("row_index")[["evidence_based", "evidence_free"]], rsuffix="_qwen")
    .join(tr_apertus_attributed.set_index("row_index")[["evidence_based", "evidence_free"]], rsuffix="_apertus")
)

# Compute averages
tr_merged_attributed["evidence_avg"] = tr_merged_attributed[["evidence_based", "evidence_based_qwen", "evidence_based_apertus"]].mean(axis=1, skipna=True)
tr_merged_attributed["intuition_avg"] = tr_merged_attributed[["evidence_free", "evidence_free_qwen", "evidence_free_apertus"]].mean(axis=1, skipna=True)
tr_merged_attributed = tr_merged_attributed.reset_index()
tr_merged_attributed["evidence_avg"] = tr_merged_attributed["evidence_avg"] / 4
tr_merged_attributed["intuition_avg"] = tr_merged_attributed["intuition_avg"] / 4
tr_merged_attributed["EMI_avg_diff_llms"] = tr_merged_attributed["evidence_avg"] - tr_merged_attributed["intuition_avg"]

tr_sbert_attributed['evidence_emb_z'] = zscore(tr_sbert_attributed['avg_evidence_score'], nan_policy='omit')
tr_sbert_attributed['intuition_emb_z'] = zscore(tr_sbert_attributed['avg_intuition_score'], nan_policy='omit')
tr_sbert_attributed['emi_embdiff'] = tr_sbert_attributed['avg_evidence_score'] - tr_sbert_attributed['avg_intuition_score']
tr_sbert_attributed['emi_embdiff_z'] = zscore(tr_sbert_attributed['emi_embdiff'], nan_policy='omit')
tr_merged_attributed["EMI_avg_diff_llms_z"] = zscore(tr_merged_attributed['EMI_avg_diff_llms'], nan_policy='omit')
tr_merged_attributed["evidence_avg_z"] = zscore(tr_merged_attributed['evidence_avg'], nan_policy='omit')
tr_merged_attributed["intuition_avg_z"] = zscore(tr_merged_attributed['intuition_avg'], nan_policy='omit')

tr_sbert_3llms_attributed = tr_sbert_attributed.join(tr_merged_attributed[['EMI_avg_diff_llms_z', 'evidence_avg_z', 'intuition_avg_z', 'chunk_uid']].set_index(['chunk_uid']),
                                  on=['chunk_uid'], how='left'
                                 )
                                 
#linear pooling
tr_sbert_3llms_attributed['rating_emb_pool'] = linear_pooling(tr_sbert_3llms_attributed['emi_embdiff_z'], tr_sbert_3llms_attributed['EMI_avg_diff_llms_z'])
tr_sbert_3llms_attributed['rating_emb_pool_z'] = zscore(tr_sbert_3llms_attributed['rating_emb_pool'], nan_policy='omit')
tr_sbert_3llms_attributed['ev_pool'] = linear_pooling(tr_sbert_3llms_attributed['evidence_avg_z'], tr_sbert_3llms_attributed['evidence_emb_z'])
tr_sbert_3llms_attributed['ev_pool_z'] = zscore(tr_sbert_3llms_attributed['ev_pool'], nan_policy='omit')
tr_sbert_3llms_attributed['int_pool'] = linear_pooling(tr_sbert_3llms_attributed['intuition_avg_z'], tr_sbert_3llms_attributed['intuition_emb_z'])
tr_sbert_3llms_attributed['int_pool_z'] = zscore(tr_sbert_3llms_attributed['int_pool'], nan_policy='omit')

boot_df_tr_attributed = bootstrap_yearly_parallel(
        df=tr_sbert_3llms_attributed,
        score_cols=score_cols,
        year_col="year",
        n_boot=10_000,
        ci=95,
        n_jobs=50,
        chunk=250,          
        batch_size=1,
        pre_dispatch="2*n_jobs",
        seed=12345,
    )
    
tr_df_yearly_attributed = (tr_sbert_3llms_attributed.groupby('year')
                             .agg(
                                 evidence_intuition_diffpool_score_mean = ('rating_emb_pool_z', 'mean'),
                                 evidence_mean                      = ('ev_pool_z', 'mean'),
                                 intuition_mean                     = ('int_pool_z', 'mean'),
                                 count                                  = ('int_pool_z', 'count'),
                                 )
                             .reset_index()
                            )

tr_ddi = tr_dem[['Year', 'Deliberative Democracy Index',                 
                 'Transparent laws with predictable enforcement', 'Clientelism Index', 'gdp_per_capita', 
                 'Judicial constraints on the executive index', 
                 ]]
tr_ddi.rename(columns = {'Year':'year', 'Deliberative Democracy Index':'ddi', 
                         'Transparent laws with predictable enforcement':'transparent_laws',
                         'Clientelism Index':'clientelism_index', 
                         'Judicial constraints on the executive index':'judiciary_index',
                         }, inplace=True
             )

tr_df_yearly_attributed = tr_df_yearly_attributed.join(tr_ddi.set_index('year'), on='year', how='left')
tr_df_yearly_attributed['country'] = 'Turkey'
tr_df_yearly_attributed.to_csv('../aggregate_data/TR/agg_sbertllms_gt1981_attributed.csv', index=False)

boot_df_tr_attributed = boot_df_tr_attributed.join(tr_ddi.set_index('year')[['ddi']], on='year', how='left')
boot_df_tr_attributed.to_csv('../aggregate_data/US/bootstrapyearly_sbertllms_gt1981.csv', index=False)



