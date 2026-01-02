
import os
import glob
import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from models.encoder import SimpleEncoder

class E5_Retriever:
    """
    A standalone retriever class for E5/Contriever models using the custom SimpleEncoder.
    """
    def __init__(
        self,
        index_dir: str,
        corpus_path: str,
        poisoned_index_dir: Optional[str] = None,
        poisoned_corpus_path: Optional[str] = None,
        model_name: str = 'intfloat/e5-large-v2',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size: int = 32
    ):
        self.device = device
        self.batch_size = batch_size
        
        print(f"Initializing E5_Retriever with model: {model_name}")
        self.encoder = SimpleEncoder(model_name, device, batch_size=batch_size)

        # 1. Load Data (Clean + Poisoned)
        clean_corpus = self._load_corpus(corpus_path)
        clean_embeddings = self._load_embeddings_from_dir(index_dir)
        self.num_clean_docs = len(clean_corpus)
        
        poison_corpus = self._load_corpus(poisoned_corpus_path) if poisoned_corpus_path else []
        poison_embeddings = self._load_embeddings_from_dir(poisoned_index_dir) if poisoned_index_dir else None

        # 2. Merge Corpora
        self.corpus = clean_corpus + poison_corpus
        print(f"Total Corpus: {len(self.corpus)} (Clean: {len(clean_corpus)}, Poisoned: {len(poison_corpus)})")

        # 3. Merge Embeddings
        all_embs = []
        if clean_embeddings is not None:
            all_embs.append(clean_embeddings)
        if poison_embeddings is not None:
            all_embs.append(poison_embeddings)
            
        if all_embs:
            self.all_embeddings = torch.cat(all_embs, dim=0).to(self.device)
            print(f"Total Index loaded. Shape: {self.all_embeddings.shape}")
        else:
            self.all_embeddings = None
            print("Warning: No embeddings loaded.")

    def _load_corpus(self, corpus_path: str) -> List[Dict]:
        corpus = []
        if not corpus_path: return corpus
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    corpus.append(json.loads(line))
        except FileNotFoundError:
            print(f"Warning: Corpus file not found at {corpus_path}.")
        return corpus

    def _load_embeddings_from_dir(self, index_dir: str) -> Optional[torch.Tensor]:
        print(f"Loading index shards from {index_dir}...")
        embeddings = []
        shard_files = sorted(glob.glob(os.path.join(index_dir, 'embedding-shard-*.pt')), 
                             key=lambda x: int(x.split('embedding-shard-')[-1].split('.pt')[0]) if '-' in x else 0)
        
        if not shard_files:
            return None
        
        for shard in shard_files:
            try:
                emb = torch.load(shard, map_location='cpu')
                embeddings.append(emb)
            except Exception as e:
                print(f"Error loading shard {shard}: {e}")

        if embeddings:
            return torch.cat(embeddings, dim=0) # Return on CPU for merging
        return None

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        # Delegate to the Clean SimpleEncoder
        return self.encoder.encode_queries(queries).to(self.device)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for the query and return top-k documents.
        Returns a list of dicts: {'id': ..., 'title': ..., 'contents': ..., 'score': ...}
        """
        if self.all_embeddings is None:
            return []

        query_emb = self.encode_queries([query]) # [1, dim]
        
        # Ensure dtypes match (safety cast)
        if self.all_embeddings is not None:
            query_emb = query_emb.to(self.all_embeddings.dtype)

        # Dot product similarity
        scores = torch.matmul(query_emb, self.all_embeddings.T).squeeze(0) # [num_docs]
        
        # Top-k
        top_k_scores, top_k_indices = torch.topk(scores, k=k)
        
        results = []
        for i in range(k):
            idx = top_k_indices[i].item()
            score = top_k_scores[i].item()
            
            doc = self.corpus[idx] if idx < len(self.corpus) else {}
            results.append({
                'id': doc.get('_id', doc.get('id', str(idx))),
                'title': doc.get('title', ''),
                'contents': doc.get('text', doc.get('contents', '')),
                'score': score,
                'is_poisoned': idx >= self.num_clean_docs
            })
            
        return results

# Helper function for quick testing
if __name__ == "__main__":
    # Example usage
    retriever = E5_Retriever(
        index_dir="datasets/hotpotqa/test_index",
        corpus_path="datasets/hotpotqa/subset_corpus.jsonl"
    )
    results = retriever.search("What is the capital of France?", k=2)
    print(json.dumps(results, indent=2))
