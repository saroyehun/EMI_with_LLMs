# shell scripts - procedural ratings
# DE
#opendiscourse data
python label_procedural_vllm.py  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"  \
    --input_csv "../data/DE/speeches_opendiscourse_bundestag_chunks_length_top100_filtered.csv" \
    --output_dir ../data/DE/opendiscourse_bundestag_procedural \
    --text_column text --language German \
    --batch_size 2048 \
    --retries 0 --id_column seq_id

# IS
python label_procedural_vllm.py  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"  \
    --input_csv ../data/IS/IGC_chunks_length_top100_filtered.csv \
    --output_dir ../data/IS/IGC_chunks_procedural_gt1945 \
    --text_column chunk_text --language Icelandic \
    --batch_size 2048 \
    --retries 0 --id_column chunk_index

# IT
python label_procedural_vllm.py  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"  \
    --input_csv "../data/IT/attributed_it_gt1945_filtertop100_chunks.csv" \
    --output_dir ../data/IT/attributed_chunks_top100filtered_gt1945_procedural \
    --text_column text --language Italian \
    --batch_size 2048 \
    --retries 0 --id_column chunk_uid

# PL
python label_procedural_vllm.py  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"  \
    --input_csv ../data/PL/tei_chunks_plenary_length_top100_filtered.csv \
    --output_dir ../data/PL/tei_chunks_procedural \
    --text_column text_chunk --language Polish \
    --batch_size 2048 \
    --retries 0 --id_column chunk_index

# TR
python label_procedural_vllm.py  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"  \
    --input_csv ../data/TR/attributed_tbmm_chunks_gt1981_filtertop100.csv \
    --output_dir ../data/TR/attributed_tbmm_chunks_gt1981_filteredtop100_procedural \
    --text_column text --language Turkish \
    --batch_size 2048 \
    --retries 0 --id_column chunk_uid

# US
python label_procedural_vllm.py  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"  \
    --input_csv "../data/US/US_congress_1946to072025_attributed_bioguide_party.csv" \
    --output_dir ../data/US/1946_2025_procedural \
    --text_column speech --language English \
    --batch_size 4096 \
    --retries 0 --id_column speech_id

