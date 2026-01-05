import argparse
import os
import json
import time
import torch
import numpy as np
from src.models import create_model
from src.utils import save_results, load_json, setup_seeds, clean_str
from src.prompts import wrap_prompt
from src.e5_retriever import E5_Retriever

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Standard RAG Attack using E5 Retriever')

    # Retriever settings
    parser.add_argument("--index_dir", type=str, default="datasets/hotpotqa/e5_index")
    parser.add_argument("--corpus_path", type=str, default="datasets/hotpotqa/corpus.jsonl")
    parser.add_argument("--poisoned_index_dir", type=str, default="datasets/hotpotqa/e5_index_poisoned")
    parser.add_argument("--poisoned_corpus_path", type=str, default="datasets/hotpotqa/poisoned_corpus.jsonl")
    parser.add_argument("--encoder_model", type=str, default="intfloat/e5-large-v2")
    
    # Evaluation settings
    parser.add_argument('--eval_dataset', type=str, default="hotpotqa")
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='palm2')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument("--query_results_dir", type=str, default='e5_attack')
    parser.add_argument("--name", type=str, default='standard_rag_attack')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = f'cuda:{args.gpu_id}'
    setup_seeds(args.seed)
    
    if args.model_config_path == None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'

    # 1. Initialize E5 Retriever (Unified Index)
    print("Initializing E5 Retriever...")
    retriever = E5_Retriever(
        index_dir=args.index_dir,
        corpus_path=args.corpus_path,
        poisoned_index_dir=args.poisoned_index_dir,
        poisoned_corpus_path=args.poisoned_corpus_path,
        model_name=args.encoder_model,
        device=device
    )

    # 2. Load Adversarial Data
    incorrect_answers = load_json(f'results/adv_targeted_results/{args.eval_dataset}.json')
    target_qids = list(incorrect_answers.keys())
    print(f"Total target samples: {len(target_qids)}")

    # 3. Initialize LLM
    llm = create_model(args.model_config_path)

    all_results = []
    asr_success_count = 0
    poisoned_retrieval_counts = []

    print(f"Starting evaluation on {len(target_qids)} questions...")

    for i, qid in enumerate(target_qids):
        print(f'############# Target Question {i+1}/{len(target_qids)}: {qid} #############')
        entry = incorrect_answers[qid]
        question = entry['question']
        target_answer = entry['incorrect answer']
        correct_answer = entry.get('correct answer', '')

        # 4. Search using Unified Retriever
        search_results = retriever.search(question, k=args.top_k)
        topk_contents = [res['contents'] for res in search_results]
        
        # Track how many poisoned docs were retrieved
        cnt_from_adv = sum([1 for res in search_results if res['is_poisoned']])
        poisoned_retrieval_counts.append(cnt_from_adv)
        
        # 5. Generate Response
        query_prompt = wrap_prompt(question, topk_contents, prompt_id=4)
        response = llm.query(query_prompt)

        # 6. Evaluate ASR
        is_asr_success = clean_str(target_answer) in clean_str(response)
        if is_asr_success:
            asr_success_count += 1

        print(f"Question: {question}")
        print(f"Output: {response}")
        print(f"ASR Success: {is_asr_success} | Poisoned Docs Retrieved: {cnt_from_adv}/{args.top_k}")
        print("-" * 30)

        all_results.append({
            "id": qid,
            "question": question,
            "topk_results": search_results,
            "input_prompt": query_prompt,
            "output": response,
            "target_answer": target_answer,
            "correct_answer": correct_answer,
            "asr_success": is_asr_success,
            "poisoned_count": cnt_from_adv
        })

    # 7. Final Summary
    total = len(target_qids)
    asr_mean = asr_success_count / total if total > 0 else 0
    poison_ratio = sum(poisoned_retrieval_counts) / (total * args.top_k) if total > 0 else 0

    print("\n" + "="*50)
    print("FINAL ATTACK RESULTS (Standard RAG)")
    print(f"Total Questions: {total}")
    print(f"ASR Mean: {asr_mean:.4f}")
    print(f"Poisoned Retrieval Ratio: {poison_ratio:.4f}")
    print("="*50)

    # 8. Save Results
    save_results(all_results, args.query_results_dir, args.name)
    print(f"Results saved to results/query_results/{args.query_results_dir}/{args.name}.json")

if __name__ == '__main__':
    main()