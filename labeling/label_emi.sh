# shell scripts - emi ratings

# DE 
python label_emi_vllm.py  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"  \
    --input_csv "../data/DE/opendiscourse_bundestag_top100procedural_filtered_gt1945.csv" \
    --output_dir ../data/DE/ \
    --text_column text --language German \
    --batch_size 2048 \
    --retries 0 --id_column seq_id

python label_emi_vllm.py  --model_name "Qwen/Qwen2.5-7B-Instruct"  \
    --input_csv "../data/DE/opendiscourse_bundestag_top100procedural_filtered_gt1945.csv" \
    --output_dir ../data/DE/ \
    --text_column text --language German \
    --batch_size 2048 \
    --retries 0 --id_column seq_id

python label_emi_vllm.py  --model_name "swiss-ai/Apertus-8B-Instruct-2509"  \
    --input_csv "../data/DE/opendiscourse_bundestag_top100procedural_filtered_gt1945.csv" \
    --output_dir ../data/DE/ \
    --text_column text --language German \
    --batch_size 2048 \
    --retries 0 --id_column seq_id
    
# IS
python label_emi_vllm.py  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"  \
   --input_csv ../data/IS/IGC_chunks_length_top100_filtered_proceduralfiltered_gt1945.csv \
    --output_dir ../data/IS/ \
    --text_column chunk_text --language Icelandic \
    --batch_size 2048 \
    --retries 0 --id_column chunk_index

python label_emi_vllm.py  --model_name "Qwen/Qwen2.5-7B-Instruct"  \
   --input_csv ../data/IS/IGC_chunks_length_top100_filtered_proceduralfiltered_gt1945.csv \
    --output_dir ../data/IS/ \
    --text_column chunk_text --language Icelandic \
    --batch_size 2048 \
    --retries 0 --id_column chunk_index

python label_emi_vllm.py  --model_name "swiss-ai/Apertus-8B-Instruct-2509"  \
   --input_csv ../data/IS/IGC_chunks_length_top100_filtered_proceduralfiltered_gt1945.csv \
    --output_dir ../data/IS/ \
    --text_column chunk_text --language Icelandic \
    --batch_size 2048 \
    --retries 0 --id_column chunk_index

# IT
python label_emi_vllm.py  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"  \
    --input_csv "../data/IT/attributed_chunks_top100proceduralfiltered_gt1945.csv" \
    --output_dir "../data/IT/" \
    --text_column text --language Italian \
    --batch_size 2048 \
    --retries 0 --id_column chunk_uid 

python label_emi_vllm.py  --model_name "Qwen/Qwen2.5-7B-Instruct"  \
    --input_csv "../data/IT/attributed_chunks_top100proceduralfiltered_gt1945.csv" \
    --output_dir "../data/IT/" \
    --text_column text --language Italian \
    --batch_size 2048 \
    --retries 0 --id_column chunk_uid 

python label_emi_vllm.py  --model_name "swiss-ai/Apertus-8B-Instruct-2509"  \
    --input_csv "../data/IT/attributed_chunks_top100proceduralfiltered_gt1945.csv" \
    --output_dir "../data/IT/" \
    --text_column text --language Italian \
    --batch_size 2048 \
    --retries 0 --id_column chunk_uid
    
# PL
python label_emi_vllm.py  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"  \
   --input_csv "../data/PL/tei_chunks_plenary_length_top100_filtered_proceduralfiltered.csv" \
    --output_dir ../data/PL/ \
    --text_column text_chunk --language Polish \
    --batch_size 2048 \
    --retries 0 --id_column chunk_index

python label_emi_vllm.py  --model_name "Qwen/Qwen2.5-7B-Instruct"  \
   --input_csv "../data/PL/tei_chunks_plenary_length_top100_filtered_proceduralfiltered.csv" \
    --output_dir ../data/PL/ \
    --text_column text_chunk --language Polish \
    --batch_size 2048 \
    --retries 0 --id_column chunk_index


python label_emi_vllm.py  --model_name "swiss-ai/Apertus-8B-Instruct-2509"  \
   --input_csv "../data/PL/tei_chunks_plenary_length_top100_filtered_proceduralfiltered.csv" \
    --output_dir ../data/PL/ \
    --text_column text_chunk --language Polish \
    --batch_size 2048 \
    --retries 0 --id_column chunk_index

# TR
python label_emi_vllm.py  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"  \
    --input_csv "../data/TR/attributed_tbmm_chunks_gt1981_filteredtop100_procedural_filtered.csv" \
    --output_dir ../data/TR/ \
    --text_column text --language Turkish \
    --batch_size 2048 \
    --retries 0 --id_column "chunk_uid"

python label_emi_vllm.py  --model_name "Qwen/Qwen2.5-7B-Instruct"  \
    --input_csv "../data/TR/attributed_tbmm_chunks_gt1981_filteredtop100_procedural_filtered.csv" \
    --output_dir ../data/TR/ \
    --text_column text --language Turkish \
    --batch_size 2048 \
    --retries 0 --id_column "chunk_uid"

python label_emi_vllm.py  --model_name "swiss-ai/Apertus-8B-Instruct-2509"  \
    --input_csv "../data/TR/attributed_tbmm_chunks_gt1981_filteredtop100_procedural_filtered.csv" \
    --output_dir ../data/TR/ \
    --text_column text --language Turkish \
    --batch_size 2048 \
    --retries 0 --id_column "chunk_uid"


# US
python label_emi_vllm.py  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct"  \
#    --input_csv "/home/jovyan/uscongress/US_congress_1946to072025_attributed_bioguide_party_proceduralllama_filtered.csv" \
#    --output_dir ../../emi_results_1946_2025 \
#    --text_column speech \
#    --batch_size 2048 \
#    --retries 0 --chunk_text 

python label_emi_vllm.py  --model_name "Qwen/Qwen2.5-7B-Instruct"  \
#    --input_csv "/home/jovyan/uscongress/US_congress_1946to072025_attributed_bioguide_party_proceduralllama_filtered.csv" \
#    --output_dir ../../emi_results_1946_2025 \
#    --text_column speech \
#    --batch_size 2048 \
#    --retries 0 --chunk_text 

python label_emi_vllm.py  --model_name "swiss-ai/Apertus-8B-Instruct-2509"  \
#    --input_csv "/home/jovyan/uscongress/US_congress_1946to072025_attributed_bioguide_party_proceduralllama_filtered.csv" \
#    --output_dir ../../emi_results_1946_2025 \
#    --text_column speech \
#    --batch_size 2048 \
#    --retries 0 --chunk_text
