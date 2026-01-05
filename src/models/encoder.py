
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from typing import List, Dict, Union, Optional
from transformers import AutoModel, AutoTokenizer

class SimpleEncoder(nn.Module):
    """
    A unified encoder for PoisonedRAG experiments.
    Supports E5 (avg pooling, prefixes) and Contriever (no prefixes usually, but supports if needed).
    """
    def __init__(
        self, 
        model_name_or_path: str, 
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_length: int = 512,
        batch_size: int = 128,
        torch_dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.model_name = model_name_or_path
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Default to float16 if on CUDA
        if torch_dtype is None and 'cuda' in device:
            torch_dtype = torch.float16
        
        self.torch_dtype = torch_dtype
        
        print(f"Initializing Unified Encoder with {model_name_or_path} on {device} (dtype: {torch_dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch_dtype
        ).to(device)
        self.model.eval()

        # Determine pooling strategy
        self.pool_type = 'cls'
        if 'e5' in model_name_or_path.lower() or 'contriever' in model_name_or_path.lower():
            self.pool_type = 'avg'
        
        print(f"Pooling Strategy: {self.pool_type}")

    def _pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pool_type == 'avg':
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pool_type == 'cls':
            return last_hidden_states[:, 0]
        else:
            return last_hidden_states[:, 0]

    def encode(self, texts: List[str], desc: str = "Encoding") -> torch.Tensor:
        all_embeddings = []
        
        iterator = range(0, len(texts), self.batch_size)
        if len(texts) > self.batch_size:
            iterator = tqdm.tqdm(iterator, desc=desc)

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_length, 
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**inputs)
                embeddings = self._pool(outputs.last_hidden_state, inputs.attention_mask)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())
        
        if len(all_embeddings) > 0:
            return torch.cat(all_embeddings, dim=0)
        else:
            return torch.tensor([])

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        # Add prefixes for E5
        if 'e5' in self.model_name.lower():
            input_texts = [f'query: {q}' for q in queries]
        else:
            input_texts = queries
            
        return self.encode(input_texts, desc="Encoding Queries")

    def encode_corpus(self, corpus: List[Dict[str, str]]) -> torch.Tensor:
        """
        Encodes a list of documents. 
        Expects dicts with 'title' and 'text' (or 'contents').
        """
        input_texts = []
        for doc in corpus:
            title = doc.get('title', '')
            # Handle both 'text' (hotpotqa) and 'contents' (wikipedia) keys
            content = doc.get('text', doc.get('contents', ''))
            
            # Format: "title \n content" or just "content"
            full_text = f"{title}\n{content}".strip()
            
            if 'e5' in self.model_name.lower():
                full_text = f'passage: {full_text}'
            
            input_texts.append(full_text)
            
        return self.encode(input_texts, desc="Encoding Corpus")
