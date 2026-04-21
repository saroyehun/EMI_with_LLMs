import pandas as pd

# TR

tr_attributed = pd.read_parquet('../data/TR/attributed_tbmm_chunks_gt1981_filteredtop100_procedural/')

len(tr_attributed)

sum(tr_attributed.procedural > 2)

tr_attributed[tr_attributed.procedural > 2].text.sample(10).tolist()

tr_attributed = tr_attributed[tr_attributed.procedural < 3]

len(tr_attributed)

tr_attributed.drop(columns=['row_index','id','error', 'partition'], inplace=True)

tr_attributed.to_csv("/home/jovyan/tr/attributed_tbmm_chunks_gt1981_filteredtop100_procedural_filtered.csv", index=False)

# PL

df_pl = pd.read_parquet("../data/PL/tei_chunks_procedural/")
len(df_pl)
sum(df_pl.procedural > 2)

df_pl[df_pl.procedural > 2].text_chunk.sample(10).tolist()

df_pl_filtered = df_pl[df_pl.procedural < 3]
len(df_pl_filtered)

df_pl_filtered.drop(columns=['row_index','id','error', 'partition'], inplace=True)

df_pl_filtered.to_csv('../data/PL/tei_chunks_plenary_length_top100_filtered_proceduralfiltered.csv', index=False)


# IS

df_is = pd.read_parquet("../data/PL/IGC_chunks_procedural_gt1945/")

sum(df_is.procedural > 2)
df_is[df_is.procedural > 2].chunk_text.sample(10).tolist()

df_is_filtered = df_is[df_is.procedural < 3]
len(df_is_filtered)

df_is_filtered.drop(columns=['row_index','id','error', 'partition'], inplace=True)

len(df_is_filtered.chunk_index.unique())

df_is_filtered.to_csv('../data/IS/IGC_chunks_length_top100_filtered_proceduralfiltered_gt1945.csv', index=False)

# DE

df_de_opendiscourse = pd.read_parquet('../data/DE/opendiscourse_bundestag_procedural')
len(df_de_opendiscourse)

df_de_opendiscourse.positionShort.value_counts(dropna=False)

sum(df_de_opendiscourse.procedural > 2)

df_de_opendiscourse[df_de_opendiscourse.procedural > 2].text.sample(10).tolist()

df_de_opendiscourse = df_de_opendiscourse[df_de_opendiscourse.procedural < 3]

len(df_de_opendiscourse)

df_de_opendiscourse.drop(columns=['row_index','id','error', 'partition'], inplace=True)

df_de_opendiscourse.to_csv('../data/DE/opendiscourse_bundestag_top100procedural_filtered_gt1945.csv',
                          index=False
                          )

# IT

df_it = pd.read_parquet('../data/IT/attributed_chunks_top100filtered_gt1945_procedural')

len(df_it)

sum(df_it.procedural > 2)

df_it[df_it.procedural > 2].text.sample(10).tolist()

df_it_filtered = df_it[df_it.procedural < 3]

len(df_it_filtered)

df_it_filtered.drop(columns=['row_index','id','error', 'partition'], inplace=True)

len(df_it_filtered.seq_id.unique())

df_it_filtered.to_csv('../data/IT/attributed_chunks_top100proceduralfiltered_gt1945.csv', index=False)
