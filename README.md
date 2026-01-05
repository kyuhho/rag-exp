# RAG-Experiments

This repository evaluates the robustness of various RAG (Retrieval-Augmented Generation) architectures against adversarial poisoning attacks.

## Environment Setup

Each architecture requires a specific environment. You can install them using the provided requirements files:

- **Standard RAG/LLM**: `pip install -r requirements_rag.txt`
- **ReAct**: `pip install -r requirements_react.txt`
- **Self-RAG**: `pip install -r requirements_selfrag.txt`
- **CoRAG**: `pip install -r requirements_corag.txt`

## Dataset Preparation

Run the following script to automate dataset download, poisoning, indexing, and mapping:

```bash
bash scripts/prepare_all_data.sh
```

## Attack Execution

Run the attack scripts for each architecture. Use `--dry_run` to test with a small number of samples.

### 1. Naive LLM (No RAG)
```bash
python attack_llm.py
```

### 2. ReAct
```bash
python ReAct/attack_react.py
```

### 3. Self-RAG
```bash
python Self-RAG/retrieval_lm/attack_selfrag.py
```

### 4. CoRAG
```bash
# Start vLLM server first
bash corag/scripts/start_vllm_server.sh

# Run attack
python corag/src/inference/attack_corag.py
```

> For Self-RAG and CoRAG, ensure you have the appropriate model servers or environment paths configured as per their respective directories.
