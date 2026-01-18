import os
import torch
import copy
import json
import threading
import logging
import sys
import re
from collections import Counter, defaultdict

# Add necessary paths to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
# 1. Add corag/src to import local modules (config, agent, etc.)
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 2. Add PoisonedRAG root to import global src (src.e5_retriever)
root_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from datasets import Dataset

from config import Arguments
from logger_config import logger
from vllm_client import VllmClient, get_vllm_model_id
from utils import save_json_to_file, AtomicCounter

from src.e5_retriever import E5_Retriever
from agent import CoRagAgent, RagPath

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


vllm_port = int(os.environ.get('VLLM_PORT', 8000))
vllm_host = os.environ.get('VLLM_HOST', 'localhost')




parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]
logger.info('Args={}'.format(str(args)))

vllm_client: VllmClient = VllmClient(
    model=get_vllm_model_id(host=vllm_host, port=vllm_port),
    host=vllm_host,
    port=vllm_port
)
corpus: Dataset = Dataset.from_dict({})

trajectory_results_dir = '/home/work/Redteaming/rag-exp/results/trajectory_results/corag'
results_path = os.path.join(trajectory_results_dir, 'clean_corag.json')

# Initialize E5_Retriever for attack
logger.info(f"Initializing E5_Retriever...")
retriever = E5_Retriever(
    index_dir="../datasets/hotpotqa/e5_index", 
    corpus_path="../datasets/hotpotqa/corpus.jsonl",
    poisoned_index_dir=None,
    poisoned_corpus_path=None,
    model_name="/home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

corag_agent: CoRagAgent = CoRagAgent(vllm_client=vllm_client, corpus=corpus, retriever=retriever)
processed_cnt: AtomicCounter = AtomicCounter()
total_cnt: int = 0
import string


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

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


def _generate_single_example(ex: Dict) -> Dict:
    # Input columns: query / query_id / answers / context_doc_ids / context_doc_scores
    # Add following columns to the output: subqueries / subanswers / path_doc_ids
    print("="*50)
    print(f"[Query] {ex['query']}")
    print("-"*50)
    if args.decode_strategy == 'greedy' or args.max_path_length < 1:
        path: RagPath = corag_agent.sample_path(
            query=ex['query'], task_desc=ex.get('task_desc', 'answer multi-hop questions'),
            max_path_length=args.max_path_length,
            temperature=0., max_tokens=64
        )
    elif args.decode_strategy == 'tree_search':
        path: RagPath = corag_agent.tree_search(
            query=ex['query'], task_desc=ex.get('task_desc', 'answer multi-hop questions'),
            max_path_length=args.max_path_length,
            temperature=args.sample_temperature, max_tokens=64
        )
    elif args.decode_strategy == 'best_of_n':
        path: RagPath = corag_agent.best_of_n(
            query=ex['query'], task_desc=ex.get('task_desc', 'answer multi-hop questions'),
            max_path_length=args.max_path_length,
            temperature=args.sample_temperature,
            n = args.best_n,
            max_tokens=64
        )
    else:
        raise ValueError(f'Invalid decode strategy: {args.decode_strategy}')

    # Collect all documents retrieved during the path sampling
    all_path_docs = []
    for docs in path.past_documents:
        all_path_docs.extend(docs)
    
    # Simple truncation if needed (optional, keeping it simple for now)
    print(f"[Final Path] {path}")
    print("-"*50)
    
    prediction: str = corag_agent.generate_final_answer(
        corag_sample=path,
        task_desc=ex.get('task_desc', 'answer multi-hop questions'),
        documents=all_path_docs,
        max_message_length=args.max_len,
        temperature=0., max_tokens=128
    )

    print(f"[Prediction] {prediction}")
    print("-"*50)

    print("="*50)

    # Save subquery and subanswer list for this question
    os.makedirs(trajectory_results_dir, exist_ok=True)
    
    # Build steps array
    steps = []
    for i in range(len(path.past_subqueries)):
        step = {
            'subquery': path.past_subqueries[i],
            'retrieved_results': path.past_retriever_results[i] if path.past_retriever_results else [],
            'subanswer': path.past_subanswers[i],
        }
        steps.append(step)
    
    question_path_data = {
        'id': ex.get('id', ''),
        'query': ex.get('query', ''),
        'correct_answer': ex.get('answers', [''])[0] if ex.get('answers') else '',
        'steps': steps,
        'final_answer': prediction,
    }
    
    # Load existing results, append new data, and overwrite file
    all_results = []
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_results = data
        except (json.JSONDecodeError, ValueError):
            all_results = []
    
    all_results.append(question_path_data)
    
    # Overwrite file with updated list
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    ex_with_path = copy.deepcopy(ex)
    ex_with_path['subqueries'] = path.past_subqueries
    ex_with_path['subanswers'] = path.past_subanswers
    ex_with_path['path_doc_ids'] = path.past_doc_ids
    if 'title' in corpus.column_names:
        ex_with_path['path_doc_titles'] = [
            [corpus[int(doc_id)]['title'] for doc_id in doc_ids] for doc_ids in path.past_doc_ids
        ]
    ex_with_path['prediction'] = prediction

    # Evaluate Accuracy (matching correct answer) - EM and F1
    correct_ans = ex.get('answers', [''])[0] if ex.get('answers') else ''
    is_accuracy_em = check_accuracy(prediction, correct_ans)
    f1, prec, recall = f1_score(prediction, correct_ans)
    is_accuracy_f1 = f1 > 0

    ex_with_path['accuracy_em'] = is_accuracy_em
    ex_with_path['accuracy_f1'] = is_accuracy_f1
    ex_with_path['f1_score'] = f1
    ex_with_path['f1_precision'] = prec
    ex_with_path['f1_recall'] = recall
    ex_with_path['correct_answer'] = correct_ans
    
    print(f"[Accuracy EM] {is_accuracy_em} | [Accuracy F1] {is_accuracy_f1} (F1={f1:.3f})")
    print(f"[Correct Ans] {correct_ans} | [Prediction] {prediction[:50]}...")

    processed_cnt.increment()
    if processed_cnt.value % 10 == 0:
        logger.info(
            f'Processed {processed_cnt.value} / {total_cnt} examples, '
            f'average token consumed: {vllm_client.token_consumed.value / processed_cnt.value:.2f}'
        )

    return ex_with_path


@torch.no_grad()
def main():
    if args.max_path_length < 1:
        logger.info('max_path_length < 1, setting decode_strategy to greedy')
        args.decode_strategy = 'greedy'
    
    with open('/home/work/Redteaming/rag-exp/ReAct/data/hotpot_dev_v1_simplified.json', 'r') as f:
        data_raw = json.load(f)
    
    # Transform adv_data to match the expected format 'ex'
    data = []
    for idx, item in enumerate(data_raw):
        data.append({
            'query': item['question'],
            'id': f"question_{idx}",
            'answer': item['answer'],
            'task_desc': 'answer multi-hop questions',
            'context_doc_ids': [],
            'context_doc_scores': []
        }) 

    # executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=args.num_threads)
    executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
    out_path: str = f'{args.output_dir}/preds_{args.decode_strategy}_{args.eval_task}_{args.eval_split}.jsonl'

    logger.info(f'Processing {args.eval_task}-{args.eval_split}...')
    # ds: Dataset = load_dataset('corag/multihopqa', args.eval_task, split=args.eval_split)
    # ds = ds.remove_columns([name for name in ['subqueries', 'subanswers', 'predictions'] if name in ds.column_names])
    # ds = ds.add_column('task_desc', ['answer multi-hop questions' for _ in range(len(ds))])

    if args.dry_run:
        data = data[:1]
    global total_cnt
    total_cnt = len(data)

    results: List[Dict] = list(executor.map(_generate_single_example, data))

    # 4. Final Summary
    total = len(results)
    accuracy_em_count = sum([1 for res in results if res.get('accuracy_em', False)])
    accuracy_f1_count = sum([1 for res in results if res.get('accuracy_f1', False)])
    avg_f1 = sum([res.get('f1_score', 0) for res in results]) / total if total > 0 else 0
    
    accuracy_em_mean = accuracy_em_count / total if total > 0 else 0
    accuracy_f1_mean = accuracy_f1_count / total if total > 0 else 0

    print("\n" + "="*50)
    print("FINAL RESULTS (CoRAG)")
    print(f"Total Questions: {total}")
    print(f"Accuracy (EM): {accuracy_em_mean:.4f} ({accuracy_em_count}/{total})")
    print(f"Accuracy (F1>0): {accuracy_f1_mean:.4f} ({accuracy_f1_count}/{total})")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print("="*50)

    # 5. Save results
    # Save summary separately
    summary_path = out_path.replace('.jsonl', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'summary': {
                'total_questions': total,
                'accuracy_em': accuracy_em_mean,
                'accuracy_em_count': accuracy_em_count,
                'accuracy_f1': accuracy_f1_mean,
                'accuracy_f1_count': accuracy_f1_count,
                'average_f1_score': avg_f1
            }
        }, f, indent=4)
    
    save_json_to_file(results, out_path, line_by_line=True)
    logger.info(f'Summary saved to {summary_path}')


if __name__ == '__main__':
    main()
