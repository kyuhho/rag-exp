import argparse
import os
import json
import time
import torch
import numpy as np
import re
import string
from collections import Counter
from src.models import create_model
from src.utils import save_results, load_json, setup_seeds, clean_str
from src.prompts import wrap_prompt
from src.e5_retriever import E5_Retriever

# HotpotQA official evaluation functions
def normalize_answer(s):
    """Normalize answer following HotpotQA official evaluation."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    if s is None:
        return ""
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def exact_match_score(prediction, ground_truth):
    """Check if prediction exactly matches ground truth after normalization."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truth):
    """Calculate F1 score following HotpotQA official evaluation."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    
    ZERO_METRIC = (0, 0, 0)
    
    # Special case: yes/no/noanswer must match exactly
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return ZERO_METRIC
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, precision, recall

def check_accuracy(prediction, correct_answer):
    """Check accuracy using HotpotQA official evaluation (EM only)."""
    if prediction is None:
        return False
    # Use exact match only
    return exact_match_score(prediction, correct_answer)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Standard RAG Attack using E5 Retriever')

    # Retriever settings
    parser.add_argument("--index_dir", type=str, default="datasets/hotpotqa/e5_index")
    parser.add_argument("--corpus_path", type=str, default="datasets/hotpotqa/corpus.jsonl")
    parser.add_argument("--poisoned_index_dir", type=str, default="datasets/hotpotqa/e5_index_poisoned")
    parser.add_argument("--poisoned_corpus_path", type=str, default="datasets/hotpotqa/poisoned_corpus.jsonl")
    parser.add_argument("--encoder_model", type=str, default="/home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6")
    
    # Evaluation settings
    parser.add_argument('--eval_dataset', type=str, default="hotpotqa")
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='llama7b')
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
    accuracy_em_count = 0
    poisoned_retrieval_counts = []
    poisoned_retrieval_flags = []  # Query당 오염문서 검색 여부 (1 or 0)

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
        # Query당 오염문서 검색 여부: 하나라도 검색되면 1, 아니면 0
        any_poisoned_retrieved = 1 if cnt_from_adv > 0 else 0
        poisoned_retrieval_flags.append(any_poisoned_retrieved)
        
        # 5. Generate Response
        query_prompt = wrap_prompt(question, topk_contents, prompt_id=4)
        response = llm.query(query_prompt)

        # 6. Evaluate ASR (Attack Success Rate - matching incorrect answer)
        is_asr_success = clean_str(target_answer) in clean_str(response)
        if is_asr_success:
            asr_success_count += 1
        
        # Evaluate Accuracy (matching correct answer) - EM and F1
        is_accuracy_em = check_accuracy(response, correct_answer)
        f1, prec, recall = f1_score(response, correct_answer)
        is_accuracy_f1 = f1 > 0
        
        if is_accuracy_em:
            accuracy_em_count += 1

        print(f"Question: {question}")
        print(f"Output: {response}")
        print(f"ASR Success: {is_asr_success} | Accuracy EM: {is_accuracy_em} | Accuracy F1: {is_accuracy_f1} (F1={f1:.3f}) | Poisoned Docs Retrieved: {cnt_from_adv}/{args.top_k}")
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
            "accuracy_em": is_accuracy_em,
            "accuracy_f1": is_accuracy_f1,
            "f1_score": f1,
            "f1_precision": prec,
            "f1_recall": recall,
            "poisoned_count": cnt_from_adv,
            "any_poisoned_retrieved": any_poisoned_retrieved
        })

    # 7. Final Summary
    total = len(target_qids)
    asr_mean = asr_success_count / total if total > 0 else 0
    accuracy_em_mean = accuracy_em_count / total if total > 0 else 0
    accuracy_f1_count = sum([1 for res in all_results if res.get('accuracy_f1', False)])
    accuracy_f1_mean = accuracy_f1_count / total if total > 0 else 0
    avg_f1 = sum([res.get('f1_score', 0) for res in all_results]) / total if total > 0 else 0
    total_poisoned_count = sum(poisoned_retrieval_counts)
    # Query당 오염문서 검색 비율: 전체 쿼리 중 하나라도 오염문서를 검색한 쿼리 비율
    poisoned_retrieval_ratio = sum(poisoned_retrieval_flags) / total if total > 0 else 0
    poisoned_queries_count = sum(poisoned_retrieval_flags)

    print("\n" + "="*50)
    print("FINAL ATTACK RESULTS (Standard RAG)")
    print(f"Total Questions: {total}")
    print(f"Attack Success Rate (ASR): {asr_mean:.4f} ({asr_success_count}/{total})")
    print(f"Accuracy (EM): {accuracy_em_mean:.4f} ({accuracy_em_count}/{total})")
    print(f"Accuracy (F1>0): {accuracy_f1_mean:.4f} ({accuracy_f1_count}/{total})")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"ASR-r: {poisoned_retrieval_ratio:.4f} ({poisoned_queries_count}/{total})")
    print(f"Total Poisoned Docs Retrieved: {total_poisoned_count}/{total * args.top_k}")
    print("="*50)

    # 8. Save Results
    # Prepare results with summary (consistent with other attack scripts)
    results_with_summary = {
        "summary": {
            "total": total,
            "asr": asr_mean,
            "asr_count": asr_success_count,
            "accuracy_em": accuracy_em_mean,
            "accuracy_em_count": accuracy_em_count,
            "accuracy_f1": accuracy_f1_mean,
            "accuracy_f1_count": accuracy_f1_count,
            "average_f1_score": avg_f1,
            "poisoned_retrieval_ratio": poisoned_retrieval_ratio,
            "poisoned_queries_count": poisoned_queries_count,
            "total_poisoned_count": total_poisoned_count,
            "total_possible_docs": total * args.top_k
        },
        "details": all_results
    }
    save_results(results_with_summary, args.query_results_dir, args.name)
    print(f"Results saved to results/query_results/{args.query_results_dir}/{args.name}.json")

if __name__ == '__main__':
    main()