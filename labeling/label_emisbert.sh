# shell scripts - emi sbert

model_name_or_path="Alibaba-NLP/gte-multilingual-base"

#IT 
python compute_sbert_avg_lexicon.py --model_name_or_path ${model_name_or_path}\
		--input_file "../data/IT/attributed_chunks_top100proceduralfiltered_gt1945.csv" \
 	--output_file "../data/IT/attributed_chunks_gt1945_sbertmgte.csv" \
        --evidence_lexicon "../data/IT/evidence_definition_IT.csv"\
        --intuition_lexicon "../data/IT/intuition_definition_IT.csv"\
        --id_column "chunk_uid" --trust_remote_code --text_column "text"

#DE 
python compute_sbert_avg_lexicon.py --model_name_or_path ${model_name_or_path}\
        --input_file "../data/DE/opendiscourse_bundestag_top100procedural_filtered_gt1945.csv"\
    	--output_file "../data/DE/opendiscourse_gt1945_top100proceduralfiltered_emi_sbertmgte.csv" \
        --evidence_lexicon "../data/DE/evidence_definition_DE.csv"\
        --intuition_lexicon "../data/DE/intuition_definition_DE.csv"\
        --id_column "seq_id" --trust_remote_code \
        --text_column text


#TR
python compute_sbert_avg_lexicon.py --model_name_or_path ${model_name_or_path}\
        --input_file "../data/TR/attributed_tbmm_chunks_gt1981_filteredtop100_procedural_filtered.csv"\
 	--output_file "../data/TR/attributed_tbmm_gt1981_filteredtop100_procedural_filtered_sbertmgte.csv" \
         --evidence_lexicon "../data/TR/evidence_definition_TR.csv"\
         --intuition_lexicon "../data/TR/intuition_definition_TR.csv"\
         --id_column "chunk_uid" --trust_remote_code --text_column text

# IS
python compute_sbert_avg_lexicon.py --model_name_or_path ${model_name_or_path}\
        --input_file "../data/IS/IGC_chunks_length_top100_filtered_proceduralfiltered_gt1945.csv"\
	--output_file "../data/IS/IGC_chunks_top100proceduralfiltered_gt1945_sbertmgte.csv" \
        --evidence_lexicon "../data/IS/evidence_definition_IS.csv"\
        --intuition_lexicon "../data/IS/intuition_definition_IS.csv"\
        --id_column "chunk_index" --trust_remote_code --text_column chunk_text

# PL
python compute_sbert_avg_lexicon.py --model_name_or_path ${model_name_or_path}\
        --input_file "../data/PL/tei_chunks_plenary_length_top100_filtered_proceduralfiltered.csv" \
	--output_file "../data/PL/tei_gt1989_sbertmgte.csv" \
        --evidence_lexicon "../data/PL/evidence_definition_PL.csv"\
        --intuition_lexicon "../data/PL/intuition_definition_PL.csv"\
        --id_column "chunk_index" --trust_remote_code --text_column text_chunk

#US
python compute_sbert_avg_lexicon.py --model_name_or_path ${model_name_or_path}\
        --input_file "../data/US/Apertus-8B-Instruct-2509_results.csv"\
	--output_file "../data/US/US_congress_gt1945_attributed_bioguide_party_procedural_chunk_sbertmgte.csv" \
        --evidence_lexicon "../data/US/evidence_definition.csv"\
        --intuition_lexicon "../data/US/intuition_definition.csv"\
        --id_column "row_index" --trust_remote_code --text_column text

