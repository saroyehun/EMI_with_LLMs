import numpy as np
import re
import os
import ast
import argparse
import pandas as pd
import random
import json
import shutil
from transformers import pipeline, AutoTokenizer
from vllm import LLM, SamplingParams, TextPrompt
from tqdm import tqdm
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=50, use_memory_fs=False)

temp_folder = "pandarallel_tmp"
os.environ["JOBLIB_TEMP_FOLDER"] = temp_folder
os.environ["PANDARALLEL_TEMP_FOLDER"] = temp_folder
os.makedirs(os.environ["PANDARALLEL_TEMP_FOLDER"], exist_ok=True)

TRUNCATE_TEXT = True
def truncate_by_words(text, max_words=300):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text

# Function to generate prompts for a batch of texts
def generate_prompts(texts, task_type, tokenizer, language='English'):
    prompts = []
    for input_text in texts:
        input_text = str(input_text).strip()
        if TRUNCATE_TEXT:
            input_text = truncate_by_words(input_text)
        prompt_template = prompt_template = f"""You are an annotator evaluating how procedural a statement is.
        
Language of the text: {language}

Definitions:
- Procedural segment: Language strictly about managing the parliamentary session or handling formal processes. This includes actions that regulate or organize the session, such as initiating or closing proceedings, enumeration of formal items (budget bills and commission reports), controlling who speaks and when, introducing or processing motions and amendments, documenting decisions, or modifying the wording of official texts. Procedural speech does not focus on the meaning or merits of the topics under discussion but on the rules and structure of how the session operates.
- Substantive segment: Any speech directed at conveying meaning, ideas, or persuasion, including debate,
  arguments, moral appeals, commemorations, or expressions of opinion. Substantive speech deals with issues, events, or people rather than the formal procedures of the session.
- Key distinction:  
Procedural = about the structure and rules of the session itself
Substantive = about the world, issues, or ideas being discussed
- Ratings are on a 0–4 scale:
0 = No procedural content at all.  
1 = Minimal procedural content within a mostly substantive statement.  
2 = Balanced mix of procedural and substantive content.  
3 = Mostly procedural with little substantive content.  
4 = Entirely procedural with no substantive content.

Instructions:
- Consider only linguistic cues in {language} when assessing whether the text segment is procedural. You never need more information than the text itself. You never need to access any external content. Always respond with a procedural rating for the text exactly as it is.
- For each statement, assign a rating for how procedural it is.
- Output must be valid JSON in the following format:
{{
  "procedural": <integer rating from 0 to 4>
}}
- Do not include any other text, explanation, or fields in the output.

""".strip()
        if task_type == "text2text-generation":
            prompt = f"Here is the Input Text: {input_text} \n" + prompt_template 
        else:
            prompt = [{"role": "system", "content": prompt_template}, {"role": "user", "content": f"Here is the Input Text: {input_text}"}]
            prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,               # return a plain string, not token IDs
                add_generation_prompt=True    # adds the generation marker at the end
                )
        prompts.append(prompt)
    return prompts

def parse_output(text):
    try:
        # Extract the JSON-like block using regex
        match = re.search(r'({.*?})', text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in output: {text}")

        json_text = match.group(1)

        # Safely parse to Python dict
        result = ast.literal_eval(json_text)

        # Ensure keys are present and cast to int
        proc_val = int(result.get("procedural", None))        

        # Validate range
        if not (0 <= proc_val <= 4):
            raise ValueError(f"Values out of range (0-4): {result}")

        return {
            "procedural": proc_val            
        }

    except (ValueError, SyntaxError, TypeError) as e:
        # Fallback: try to recover scores from messy output
        fallback_match = re.search(
            r'"?procedural"?\s*:\s*(\d+)',
            text,
            re.DOTALL
        )
        if fallback_match:
            proc_val = int(fallback_match.group(1))            

            if 0 <= proc_val <= 4:
                return {
                     "procedural": proc_val
                }
        raise ValueError(f"Failed to parse model output: {text}\nError: {e}")


# Function to parse outputs for a batch
def parse_batch_outputs(outputs, texts, ids, task_type):
    results = []
    for output, text, id_ in zip(outputs, texts, ids):
        try:
            response = output.outputs[0].text.strip()
            parsed_output = parse_output(response)
            results.append({
                "id": id_,
                "text": text,
                "procedural": parsed_output["procedural"],                
                "error": None
            })

        except (ValueError, SyntaxError, AttributeError) as e:
            results.append({
                "id": id_,
                "text": text,
                "procedural": None,                
                "error": str(e)
            })

    return results


# Function to retry invalid responses within a batch
def retry_invalid_responses(model, results, retries, task_type, tokenizer):
    for attempt in range(retries):
        invalid_results = [
            r for r in results
            if r["procedural"] is None
        ]
        if not invalid_results:
            break  
     
        texts_to_retry = [r["text"] for r in invalid_results]
        ids_to_retry = [r["id"] for r in invalid_results]

        prompts = generate_prompts(texts_to_retry, task_type, tokenizer)
        sampling_params = SamplingParams(
            max_tokens=50,             # similar to max_new_tokens            
        )
        outputs = model.generate(prompts, sampling_params)

        new_results = parse_batch_outputs(outputs, texts_to_retry, ids_to_retry, task_type)
       
        for original, new in zip(invalid_results, new_results):
            if new["procedural"] is not None:
                original.update(new)
            else:
                original["error"] = f"Retry {attempt + 1}: {new['error']}"

    return results

# Main batch processing function
def process_batches(model_name, texts, ids, batch_size=8, retries=3, task_type="text-generation", trust_remote_code=False, language="English"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_generator = LLM(model=model_name, dtype="bfloat16",trust_remote_code=trust_remote_code,
                        max_model_len=4096,          
                        gpu_memory_utilization=0.7,  # use only 70% of VRAM                        
                        )
    
    # Define generation parameters
    sampling_params = SamplingParams(
        max_tokens=50,             # similar to max_new_tokens  
    )
    
    all_results = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing batches with {model_name}"):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        prompts = generate_prompts(batch_texts, task_type, tokenizer, language)
        outputs = text_generator.generate(prompts, sampling_params)
        results = parse_batch_outputs(outputs, batch_texts, batch_ids, task_type)
        if retries > 0:
            results = retry_invalid_responses(text_generator, results, retries, task_type, tokenizer)
        all_results.extend(results)

    return all_results

def make_parquet_safe(df):
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype("string")
    return df

# Command-line interface with argparse
def main():
    parser = argparse.ArgumentParser(description="Run emotion and sentiment prediction with an 8B LLM.")
    parser.add_argument("--model_name", required=True, help="The name of the HuggingFace model to use.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file containing text data.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output CSV file(s).")
    parser.add_argument("--text_column", default="text", help="Column name containing input text in the CSV file.")
    parser.add_argument("--compression", help="Compression type of the CSV file.")
    parser.add_argument("--task_type", default="continuation", help="Type of task, eeither of text-generation or text2text-generation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing.")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for invalid outputs.")
    parser.add_argument("--trust_remote_code", action='store_true', help="Whether to trust remote code or not.")
    parser.add_argument("--smoke_test", action='store_true', help="Whether to do a quick run.")
    parser.add_argument("--chunk_text", action='store_true', help="Whether to chunk text.")
    parser.add_argument('--min_chunk_length', type=int, default=50 )
    parser.add_argument('--max_chunk_length', type=int, default=150)
    parser.add_argument('--id_column', type=str, default="speech_id")
    parser.add_argument('--language', type=str, default="English")
    parser.add_argument("--chunksize", type=int, default=1_000_000, 
                        help="Number of rows to process at a time."
                       )
    parser.add_argument("--output_parquet", type=str, help="Path to the output file.")

    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model_name = args.model_name.split('/')[-1]
    # Load input texts
    compression = args.compression if args.compression else None
    nrows = 100 if args.smoke_test else None
    with pd.read_csv(args.input_csv, chunksize=args.chunksize, 
                     compression=args.compression, nrows=nrows,                     
                    ) as reader:
        for chunk_idx, chunk in tqdm(enumerate(reader)):
            df = chunk.drop_duplicates(subset=[args.id_column, args.text_column])
            if 'speaker' in df.columns:
                df = df[~df.speaker.fillna('').astype(str).str.lower().str.contains(r"\bspeaker\b|\bclerk\b", na=False)]
    
            if args.text_column not in df.columns:
                raise ValueError(f"Column '{args.text_column}' not found in the input CSV.")    
            if args.chunk_text:
                def chunk_by_length(x):
                    max_chunk_length = args.max_chunk_length
                    words = x.split()
                    if len(words) > max_chunk_length:
                        chunks = [words[i:i+max_chunk_length] for i in range(0, len(words), max_chunk_length)]
                        last_chunk_length = len(chunks[-1])
                        if len(chunks) > 1 and last_chunk_length < args.min_chunk_length:
                            chunks[-2] = chunks[-2] + chunks[-1]
                            del chunks[-1]
                        chunked = [" ".join(chunk) for chunk in chunks]
                    else:
                        chunked = [" ".join(words)]
                    return chunked 
        
                batch_size = 500_000
                num_splits = (len(df) // batch_size) + 1
                splits = np.array_split(df, num_splits)
                
                def dedup_list(chunks):
                    return list(dict.fromkeys(chunks))
                    
                def process_batch(batch, args):
                    batch['text'] = batch[args.text_column].parallel_apply(chunk_by_length)
                    batch['text'] = batch['text'].parallel_apply(dedup_list)
                    if args.text_column != 'text':
                        batch.drop(columns=[args.text_column], inplace=True)
                    batch = batch.explode('text', ignore_index=True)
                    batch = batch.drop_duplicates(subset=['text', args.id_column])
                    return batch
        
                processed_batches = []
                for i, batch in tqdm(enumerate(splits)):
                    #print(f"Processing batch {i+1}/{num_splits} with {len(batch)} rows...")
                    processed = process_batch(batch.copy(), args)
                    #processed.to_parquet(f"processed_batch_{i}.parquet")
                    processed_batches.append(processed)
        
                df = pd.concat(processed_batches, ignore_index=True)
                df = df.drop_duplicates(subset=['text', args.id_column])
                
            df.reset_index(drop=False, inplace=True)
            if "row_index" not in df.columns:
                df.rename(columns={"index": "row_index"}, inplace=True)
            ids = df["row_index"].tolist()
            text_column = 'text' if args.chunk_text else args.text_column
            df[text_column] = df[text_column].astype(str)
            texts = df[text_column].tolist()
        
            # Process texts
            results = process_batches(args.model_name, texts, ids, batch_size=args.batch_size, 
                                      retries=args.retries, task_type=args.task_type,
                                      trust_remote_code=args.trust_remote_code,
                                      language=args.language
                                     )
            results_df = pd.DataFrame(results)
            results_df = df.merge(results_df[['id', 'procedural', 'error']],
                                  left_on="row_index", right_on="id", 
                                  how="left"
                                 )
            results_df['partition'] = chunk_idx
            results_df[args.id_column] = results_df[args.id_column].astype(str)
            results_df = make_parquet_safe(results_df)
            results_df.to_parquet(f"{args.output_dir}/{model_name}.parquet",
                                  index=False, partition_cols=['partition'],
                                  engine='pyarrow')

   
    temp_folder = os.environ.get("PANDARALLEL_TEMP_FOLDER", None)
    if temp_folder and os.path.exists(temp_folder):
        print(f"Cleaning Pandarallel temp folder: {temp_folder}")
        shutil.rmtree(temp_folder)

if __name__ == "__main__":
    main()
