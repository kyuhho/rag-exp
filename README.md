# PoisonedRAG

This repository evaluates the robustness of various RAG (Retrieval-Augmented Generation) architectures against adversarial poisoning attacks.

## 🛠 Dataset Preparation

Run the following script to automate dataset download, poisoning, indexing, and mapping:

```bash
bash scripts/prepare_all_data.sh
```

## 🚀 Attack Execution

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
python corag/src/inference/attack_corag.py
```

> [!NOTE]
> For Self-RAG and CoRAG, ensure you have the appropriate model servers or environment paths configured as per their respective directories.
