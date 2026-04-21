from sentence_transformers import SentenceTransformer, util, models
from sentence_transformers import CrossEncoder
import matplotlib.pyplot as plt
import sys
import re
import os
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from scipy.stats import zscore
import string
import shutil
from wordfreq import top_n_list
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=50, use_memory_fs=False)

temp_folder = "pandarallel_tmp"
os.environ["JOBLIB_TEMP_FOLDER"] = temp_folder
os.environ["PANDARALLEL_TEMP_FOLDER"] = temp_folder
os.makedirs(os.environ["PANDARALLEL_TEMP_FOLDER"], exist_ok=True)

def config(parser):
    parser.add_argument('--model_name_or_path')
    parser.add_argument('--input_file')
    parser.add_argument('--output_file')
    parser.add_argument('--evidence_lexicon')
    parser.add_argument('--intuition_lexicon')
    parser.add_argument('--save_embeddings', action="store_true")
    parser.add_argument('--smoke_test', action="store_true")
    parser.add_argument('--text_column', type=str, default='text')
    parser.add_argument('--compression_type', type=str, default='infer')
    parser.add_argument('--length_threshold', type=int, default=10)
    parser.add_argument('--tab_delimiter', action="store_true")
    parser.add_argument('--chunk_text', action="store_true")
    parser.add_argument('--min_chunk_length', type=int, default=50 )
    parser.add_argument('--max_chunk_length', type=int, default=150)
    parser.add_argument('--id_column', type=str, default="speech_id")
    parser.add_argument('--start_year', type=int,)
    parser.add_argument('--trust_remote_code', action="store_true")
    parser.add_argument('--preprocess', action="store_true")
    return parser 


def get_embeddings(text, model):
    #encode text in batches 
    corpus_embeddings = model.encode(text, batch_size=8, show_progress_bar=True, convert_to_tensor=True)
    assert len(corpus_embeddings) == len(text)
    return corpus_embeddings

top100 = top_n_list('en', 100)
def count_top100(s):
    return len([1 for w in s.split() if w in set(top100)])

def preprocess(df, args):
    #replace multiple occurrence of .
    df.text.replace(to_replace=r"\.\.+", value=" ", regex=True, inplace=True)
    df.text.replace(to_replace=r"\-\-+", value=" ", regex=True, inplace=True)
    df.text.replace(to_replace=r"__+", value=" ", regex=True, inplace=True)
    df.text.replace(to_replace=r"\*\*+", value=" ", regex=True, inplace=True)
    df.text.replace(to_replace=r"\s+", value=" ", regex=True, inplace=True)
    df['length'] = df.text.parallel_apply(lambda x: len(x.split()))
    df = df[df.length > args.length_threshold]
    print(len(df))

    df['tokens_top100'] = df.text.parallel_apply(count_top100)
    df['fraction_top100'] = df.tokens_top100 / df.length
    try:
        print('sample top100 ', df[(df.fraction_top100 < 0.05)].sample(10).text.tolist())
        df = df[~(df.fraction_top100 < 0.05)] 
        print(len(df))
    except:
        print('nothing to sample')
    
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

        df['text'] = df.text.parallel_apply(chunk_by_length)
        df = df.explode("text", ignore_index=True)    
        df = df.drop_duplicates(subset=['text']+[f'{args.id_column}'])
        df['chunk_length'] = df.text.parallel_apply(lambda x: len(x.split()))
    return df

def main(args):
    delimiter = '\t' if args.tab_delimiter else None
    if args.smoke_test:
        df = pd.read_csv(args.input_file, nrows=10_000, compression=args.compression_type, delimiter=delimiter, dtype={'speech_id':object})
    else:
        df = pd.read_csv(args.input_file, compression=args.compression_type, delimiter=delimiter, dtype={'speech_id':object})
    #rename text column if different from text
    df = df.drop(df.filter(regex='evidence', axis=1).columns, axis=1)
    df = df.drop(df.filter(regex='intuition', axis=1).columns, axis=1)
    if args.start_year:
        df = df[df['year'] >= args.start_year]
    if args.text_column != 'text':
        if 'text' in df.columns:
            df = df.drop(columns=['text'])
        df.rename(columns = {args.text_column:'text'}, inplace = True)
    df['text'] = df['text'].astype(str)
    df = df.drop_duplicates(subset=['text']+[f'{args.id_column}'])
    print('Before pre-processing:', len(df))
    if args.preprocess:
        df = preprocess(df, args)
        print('After pre-processing:', len(df[f'{args.id_column}'].unique()))

    model = SentenceTransformer(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    model.max_seq_length = 512
    evidence_sim = torch.Tensor()
    intuition_sim = torch.Tensor()

    chunk_size = 1_000_000
    list_df = [df[idx:idx+chunk_size] for idx in range(0, len(df), chunk_size)]
    for batch in tqdm(list_df):
        batch_text = batch['text']
        batch_text = list(batch_text)

        evidence_keywords = pd.read_csv(args.evidence_lexicon) 
        evidence_keywords = list(evidence_keywords['evidence_keywords'])  
        intuition_keywords = pd.read_csv(args.intuition_lexicon) 
        intuition_keywords = list(intuition_keywords['intuition_keywords'])
        
        text_embeddings = get_embeddings(batch_text, model)       

        if args.save_embeddings:
            import pickle 
            output_fn = args.output_file.replace(".csv", ".pkl")
            with open(output_fn, "wb") as fout:
                pickle.dump({'text': all_text, 'embeddings': text_embeddings}, fout, protocol=pickle.HIGHEST_PROTOCOL)

        evidence_embeddings = get_embeddings(evidence_keywords, model)
        evidence_embeddings = torch.mean(evidence_embeddings, dim=0)

        intuition_embeddings = get_embeddings(intuition_keywords, model)
        intuition_embeddings = torch.mean(intuition_embeddings, dim=0)

        evidence_sim = torch.cat((evidence_sim, util.cos_sim(text_embeddings, evidence_embeddings).cpu()), 0)
        intuition_sim = torch.cat((intuition_sim, util.cos_sim(text_embeddings, intuition_embeddings).cpu()), 0)

    avg_evidence_score = np.average(evidence_sim.cpu().numpy(), axis=1)  
    avg_intuition_score = np.average(intuition_sim.cpu().numpy(), axis=1) 
                
    df['avg_evidence_score'] = (avg_evidence_score + 1) / 2
    df['avg_intuition_score'] = (avg_intuition_score + 1) / 2
    
    df['evidence_minus_intuition_score'] = df['avg_evidence_score'] - df['avg_intuition_score']
    print(df.evidence_minus_intuition_score.head())
    print(df.evidence_minus_intuition_score.tail())
    
    out_dir = os.path.dirname(os.path.abspath(args.output_file))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output_file, index=False, compression=args.compression_type)
    
    temp_folder = os.environ.get("PANDARALLEL_TEMP_FOLDER", None)
    if temp_folder and os.path.exists(temp_folder):
        print(f"Cleaning Pandarallel temp folder: {temp_folder}")
        shutil.rmtree(temp_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = config(parser)
    args = parser.parse_args()
    main(args)
