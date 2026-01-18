
import os
import sys
import argparse
import json
import torch
import tqdm
from typing import List, Dict

# Add src to path to import models.encoder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from models.encoder import SimpleEncoder
from logger_config import logger

def load_jsonl(path: str) -> List[Dict]:
    logger.info(f"Loading corpus from {path}...")
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} documents.")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, required=True, help='Path to corpus.jsonl')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save .pt shards')
    parser.add_argument('--model_name', type=str, default='intfloat/e5-large-v2', help='Encoder model name')
    parser.add_argument('--batch_size', type=int, default=1024, help='Inference batch size')
    parser.add_argument('--shard_size', type=int, default=100000, help='Number of documents per shard')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Corpus
    corpus = load_jsonl(args.corpus_path)

    # Initialize Encoder
    logger.info(f"Initializing Encoder: {args.model_name}")
    encoder = SimpleEncoder(model_name_or_path=args.model_name, device=args.device, batch_size=args.batch_size)

    # Encoding Loop
    logger.info(f"Starting encoding with shard size {args.shard_size}...")
    
    total_docs = len(corpus)
    for shard_idx, start_idx in enumerate(range(0, total_docs, args.shard_size)):
        end_idx = min(start_idx + args.shard_size, total_docs)
        shard_corpus = corpus[start_idx:end_idx]
        
        logger.info(f"Encoding shard {shard_idx}: docs {start_idx} to {end_idx}")
        
        # SimpleEncoder.encode_corpus expects a list of dicts with 'title' and 'contents'/'text'
        # Ensure compatibility if keys differ
        normalized_corpus = []
        for doc in shard_corpus:
            norm_doc = {
                'title': doc.get('title', ''),
                'contents': doc.get('contents', doc.get('text', ''))
            }
            normalized_corpus.append(norm_doc)

        embeddings = encoder.encode_corpus(normalized_corpus)
        
        # Save Shard
        save_path = os.path.join(args.output_dir, f'embedding-shard-{shard_idx}.pt')
        torch.save(embeddings, save_path)
        logger.info(f"Saved shard {shard_idx} to {save_path}")

    logger.info("Indexing completed successfully.")

if __name__ == '__main__':
    main()
