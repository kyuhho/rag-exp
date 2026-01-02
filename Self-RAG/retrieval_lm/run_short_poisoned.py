#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import json
import logging
import os
import random
import sys
import copy
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# [FIXME] Force root import by confusing sys.path
# Python adds script dir to sys.path[0] automatically, causing local 'src' to shadow root 'src'.
# We must remove it temporarily.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))

if current_dir in sys.path:
    sys.path.remove(current_dir)
sys.path.insert(0, root_dir)

# Now we can import from root src
from src.utils import load_models, clean_str
from src.attack import Attacker

# Restore local path for local imports
if root_dir in sys.path:
    sys.path.remove(root_dir) # Optional cleanup
sys.path.insert(0, current_dir)
sys.path.append(root_dir) # Keep root at end just in case

# [REFERENCE] Importing utility functions from run_short_form.py dependencies
# Ensure this script is run from Self-RAG/retrieval_lm or PYTHONPATH is set correctly
from utils import PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens, load_special_tokens
from metrics import match, accuracy

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

seed = 633
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# [REFERENCE] Helper functions from run_short_form.py
def postprocess_answer_option_conditioned(answer):
    for token in control_tokens:
        answer = answer.replace(token, "")
    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")
    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")
    return answer

def call_model_rerank_w_scores_batch(prompt, evidences, model, max_new_tokens=15,
                                     ret_tokens=None, rel_tokens=None, grd_tokens=None, ut_tokens=None,
                                     use_seqscore=False, threshold=0.5,
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=False):
    # This logic remains identical to run_short_form.py
    results = {}
    if mode != "always_retrieve":
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=32016)
        preds = model.generate([prompt], sampling_params)
        pred_token_ids = preds[0].outputs[0].token_ids
        pred_text = preds[0].outputs[0].text
        pred_log_probs = preds[0].outputs[0].logprobs
        results["no_retrieval"] = pred_text

    if mode == "always_retrieve":
        do_retrieve = True
    elif mode == "no_retrieval":
        do_retrieve = False
    else:
        if threshold is not None:
            score_dict = {}
            for tok, id in ret_tokens.items():
                if id not in pred_log_probs[0]:
                    score_dict[tok] = -100
                prob = pred_log_probs[0][id]
                score_dict[tok] = float(prob)
            do_retrieve = score_dict["[Retrieval]"] / (
                score_dict["[Retrieval]"] + score_dict["[No Retrieval]"]) > threshold
        else:
            do_retrieve = "[Retrieval]" in pred

    if do_retrieve is True:
        evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(
            para["title"], para["text"]) for para in evidences]
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=5000)
        preds = model.generate(evidence_augmented_inputs, sampling_params)

        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        for p_idx, pred in enumerate(preds):
            pred_token_ids = pred.outputs[0].token_ids
            pred_text = pred.outputs[0].text
            pred_log_probs = pred.outputs[0].logprobs
            seq_score = pred.outputs[0].cumulative_logprob / max(len(pred.outputs[0].token_ids), 1)

            relevance_score_dict.setdefault(p_idx, {})
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            
            for tok, id in rel_tokens.items():
                prob = pred_log_probs[0][id] if id in pred_log_probs[0] else -100
                relevance_score_dict[p_idx][tok] = np.exp(float(prob))

            if grd_tokens is not None:
                groundness_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(grd_tokens.values()):
                        groundness_token_appear_indices.append(tok_idx)
                        break
                if len(groundness_token_appear_indices) > 0:
                    idx = groundness_token_appear_indices[0]
                    for token, token_id in grd_tokens.items():
                        prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                        grd_score_dict[p_idx][token] = np.exp(float(prob))

            if ut_tokens is not None:
                utility_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(ut_tokens.values()):
                        utility_token_appear_indices.append(tok_idx)
                if len(utility_token_appear_indices) > 0:
                    idx = utility_token_appear_indices[0]
                    for token, token_id in ut_tokens.items():
                        prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                        ut_score_dict[p_idx][token] = np.exp(float(prob))

            relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (
                np.sum(list(relevance_score_dict[p_idx].values())))

            if len(grd_score_dict[p_idx]) == 3:
                gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
                ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (
                    grd_score_dict[p_idx]["[Partially supported]"] / gt_sum)
            else:
                ground_score = 0.0

            if len(ut_score_dict[p_idx]) == 5:
                ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
                ut_scores = [-1, -0.5, 0, 0.5, 1]
                utility_score = np.sum(
                    [ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum) for i in range(len(ut_scores))])
            else:
                utility_score = 0.0

            if use_seqscore is True:
                final_score = np.exp(seq_score) + w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score
            else:
                final_score = w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score

            overall_scores[p_idx] = {"final_score": final_score,
                                     "relevance_score": relevance_score,
                                     "ground_score": ground_score,
                                     "utility_score": utility_score,
                                     "relevance_score_dict": relevance_score_dict,
                                     "grd_score_dict": grd_score_dict,
                                     "ut_score_dict": utility_score}
            results["retrieval_{}".format(p_idx)] = {
                "pred": pred_text, "score": final_score, "ctx": evidences[p_idx]}

    else:
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
        prompt += "[No Retrieval]"
        preds = model.generate([prompt], sampling_params)
        pred = preds[0].outputs[0].text

    if len(results) == 1:
        postprocessed_pred = postprocess_answer_option_conditioned(pred)
        return postprocessed_pred, results, do_retrieve
    else:
        answer2score = {}
        if closed is True:
            for key, result in results.items():
                if key == "no_retrieval": continue
                answer = postprocess_answer_option_conditioned(result["pred"])
                score = result["score"]
                answer2score.setdefault(answer, 0)
                answer2score[answer] += score
            sorted_answers = sorted(answer2score.items(), key=lambda x: x[1], reverse=True)
            best_option = sorted_answers[0][0]
        else:
            path2score = {key: item["score"] for key, item in results.items() if key != "no_retrieval"}
            best_path = sorted(path2score.items(), key=lambda x: x[1], reverse=True)[0][0]
            best_option = results[best_path]["pred"]
        return best_option, results, do_retrieve

def process_data_evidences(demonstration, top_n):
    ctx_key = "ctxs" if "ctxs" in demonstration else "top_contexts"
    prompt = PROMPT_DICT["prompt_no_input"].format_map(demonstration)
    evidences = demonstration[ctx_key][:top_n]
    return prompt, evidences

def preprocess_input_data(dataset, task=None):
    new_data = []
    if task in TASK_INST:
        instruction = TASK_INST[task]
    else:
        instruction = None
    for item in dataset:
        prompt = instruction + "\n\n## Input:\n\n" + item["question"] if instruction is not None else item["question"]
        item["instruction"] = prompt
        new_data.append(item)
    return new_data

# [MODIFIED] Helper Function to get embeddings using Contriever (same as src/attack.py)
def get_emb(model, input_dict):
    with torch.no_grad():
        return model(**input_dict)

# [MODIFIED] Data Construction Logic to support FAIR, COMPETITIVE POISONING
def construct_hotpotqa_poisoned_data(queries_path, corpus_path, beir_results_path, adv_results_path, ndocs, 
                                     ret_model, ret_tokenizer, device, attacker, args): # Now requires attacker and args
    logger.info("Constructing Poisoned HotpotQA dataset with FAIR SCORING (Competition Mode)...")
    
    # 1. Load Adversarial Results (Candidates)
    logger.info(f"Loading Adversarial Texts from {adv_results_path}...")
    with open(adv_results_path, 'r') as f:
        adv_data_raw = json.load(f)
        
    target_qids = set()
    qid_to_adv = {}
    if isinstance(adv_data_raw, dict):
         for qid, data in adv_data_raw.items(): # FIXME: check logic (Iterate ADVs to determine TARGETS)
             target_qids.add(qid)
             if 'adv_texts' in data:
                 # Store ALL adversarial candidates for competition
                 qid_to_adv[str(qid)] = {
                     'candidates': data['adv_texts'], # [ATTACK] Store all 5 candidates
                     'incorrect_answer': data.get('incorrect answer', '')
                 }

    logger.info(f"Identified {len(target_qids)} Target Queries for Attack.")

    # 2. Load BEIR results (Mapping QueryID -> {DocID: Score})
    logger.info(f"Loading BEIR results from {beir_results_path}...")
    with open(beir_results_path, 'r') as f:
        beir_results = json.load(f)
    
    # 3. Collect DocIDs
    # 모든 target query에 대한 상위 ndocs개의 문서 id set 저장
    required_doc_ids = set()
    # 각 query id 당 상위 ndocs개씩 저장
    query_to_top_docs = {}
    for qid in target_qids: # FIXME: check logic (Filtering optimization)
        if qid in beir_results:
            results = beir_results[qid]
            # Take top N original candidates to compete against
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:ndocs]
            # Store pairs (doc_id, score) for easier merging
            query_to_top_docs[qid] = sorted_results 
            for doc_id, _ in sorted_results:
                required_doc_ids.add(doc_id)
            
    logger.info(f"Identified {len(required_doc_ids)} unique documents needed from corpus.")

    # 4. Load Corpus
    logger.info(f"Loading Corpus from {corpus_path}...")
    doc_lookup = {}
    with open(corpus_path, 'r') as f:
        first_line = json.loads(f.readline())
        id_key = '_id' if '_id' in first_line else 'id'
        f.seek(0)
        
        for line in tqdm(f):
            item = json.loads(line)
            if item[id_key] in required_doc_ids: # FIXME: check logic (Load only needed docs)
                doc_lookup[item[id_key]] = item 
    # 5. Load Queries and Perform FAIR COMPETITION
    logger.info(f"Loading Queries and calculating scores (This may take a while)...")
    final_dataset = []
    
    # Move model to device just in case
    ret_model.to(device)
    ret_model.eval()

    poisoned_cnt = 0
    total_cnt = 0

    # [NEW] Pre-collect target queries for batch attack generation
    # main.py format: list of dicts {'query': q, 'top1_score': score, 'id': id}
    target_query_objects = []
    
    # Needs Top-1 Score for Attacker (main.py line 113)
    # We will fetch this from beir_results
    
    with open(queries_path, 'r') as f:
        # Read file completely first to map ID to Text (for efficiency)
        # Or read line by line. Given file size, let's just loop.
        # But Attacker needs ALL targets at once.
        # So we scan file to find target query texts.
        
        # Helper map: qid -> text
        qid_to_text = {}
        first_line = json.loads(f.readline())
        id_key = '_id' if '_id' in first_line else 'id'
        text_key = 'text' if 'text' in first_line else 'question'
        f.seek(0)
        
        for line in f:
            item = json.loads(line)
            qid = str(item[id_key])
            if qid in target_qids:
                qid_to_text[qid] = item[text_key]

    # Construct target_queries list for Attacker
    for qid in target_qids:
        # Get Top-1 Score from BEIR
        if qid in beir_results:
             # beir_results[qid] is {docid: score}
             # Max score
             top1_score = max(beir_results[qid].values())
             target_query_objects.append({
                 'query': qid_to_text[qid],
                 'top1_score': top1_score,
                 'id': qid
             })

    logger.info(f"Generating Dynamic Attacks for {len(target_query_objects)} queries using Attacker (Simulating main.py)...")
    # [ATTACK] Dynamic Generation
    # attacker.get_attack returns list of lists (groups of adv texts)
    adv_text_groups = attacker.get_attack(target_query_objects)
    
    # Map back to QID for easy lookup
    # Order matches target_query_objects
    qid_to_generated_adv = {}
    for idx, obj in enumerate(target_query_objects):
        qid = obj['id']
        qid_to_generated_adv[qid] = adv_text_groups[idx]

    with open(queries_path, 'r') as f:
        f.seek(0) # Reset
        
        for line in tqdm(f, total=len(target_qids)): # Approximate total
            item = json.loads(line)
            qid = str(item[id_key])
            question = item[text_key]
            # [MODIFIED] STRICT FILTER
            if qid not in target_qids: # FIXME: check logic (Strict Target Filtering)
                continue
            if qid not in query_to_top_docs:
                continue
    
            # 5a. Prepare Original Top-K Candidates
            candidates = [] # list of dicts: {title, text, score}
            
            for doc_id, original_score in query_to_top_docs[qid]:
                if doc_id in doc_lookup:
                    doc = doc_lookup[doc_id]
                    candidates.append({
                        "id": str(doc.get(id_key)),
                        "title": doc.get('title', ''),
                        "text": doc.get('text', ''),
                        "score": float(original_score) # Original Score from BEIR (Dot Product)
                    })
            
            # 5b. Prepare Adversarial Candidates and Calculate Scores
            # [ATTACK] Here we calculate scores for ALL adversarial texts and let them COMPETE
            # 5b. Prepare Adversarial Candidates and Calculate Scores
            # [ATTACK] Use DYNAMICALLY GENERATED attacks
            if qid in qid_to_generated_adv: # FIXME: check logic (FAIR COMPETITION LOGIC implementation)
                adv_candidates_text = qid_to_generated_adv[qid]
                # incorrect answer needed for metric?
                # We can grab it from loading adv_file earlier (qid_to_adv)
                # qid_to_adv[qid]['incorrect_answer'] is still valid since we loaded it from json
                incorrect_answer = qid_to_adv[qid]['incorrect_answer']
                
                # Encode Query
                q_input = ret_tokenizer(question, padding=True, truncation=True, return_tensors="pt")
                q_input = {key: value.to(device) for key, value in q_input.items()}
                with torch.no_grad():
                    q_emb = get_emb(ret_model, q_input) # Shape: [1, Dim]

                # Encode Adv Candidates
                adv_input = ret_tokenizer(adv_candidates_text, padding=True, truncation=True, return_tensors="pt")
                adv_input = {key: value.to(device) for key, value in adv_input.items()}
                with torch.no_grad():
                    adv_embs = get_emb(ret_model, adv_input) # Shape: [N_adv, Dim]

                # Iterate one by one to match main.py's Vector-Vector calculation order
                for j in range(len(adv_embs)):
                    adv_emb = adv_embs[j, :].unsqueeze(0) # [1, Dim]
                    # Direct replica of main.py line 163:
                    # adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
                    score = torch.mm(adv_emb, q_emb.T).cpu().item()
                    
                    candidates.append({
                        "id": f"poison_{j}",
                        "title": "Adversarial Passage",
                        "text": adv_candidates_text[j],
                        "score": score
                    })
                
                item['target_incorrect_answer'] = incorrect_answer
            
            # 5c. Sort All Candidates by Score (Competition)
            # [ATTACK] Sort reverse=True (Higher score is better)
            candidates.sort(key=lambda x: x['score'], reverse=True)

            # 5d. Take Top-K
            final_ctxs = candidates[:ndocs]

            poisoned_candidates = [c for c in final_ctxs if c['id'].startswith('poison')]
            poisoned_cnt += len(poisoned_candidates)
            total_cnt += len(final_ctxs)
            
            data_item = {
                "id": qid,
                "question": question,
                "answers": [], 
                "ctxs": final_ctxs,
                "target_incorrect_answer": item.get('target_incorrect_answer', '') # FIXME: check logic (Added target mismatch tracking)
            }
            final_dataset.append(data_item)
            
    logger.info(f"Constructed {len(final_dataset)} poisoned data items via Fair Competition.")
    logger.info(f"Poisoned Ratio: {poisoned_cnt / total_cnt} {poisoned_cnt} / {total_cnt}")
    return final_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="selfrag/selfrag_llama2_7b")
    # New arguments for data construction
    # FIXME: check logic (New Arguments added for separate data files)
    parser.add_argument('--queries_file', type=str, required=True, help="Path to queries.jsonl")
    parser.add_argument('--corpus_file', type=str, required=True, help="Path to corpus.jsonl")
    parser.add_argument('--beir_results', type=str, required=True, help="Path to beir results json")
    parser.add_argument('--adv_file', type=str, required=True, help="Path to adversarial texts json")
    
    parser.add_argument('--output_file', type=str, default="hotpotqa_poisoned_results.json")
    parser.add_argument('--task', type=str, default="hotpotqa")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--download_dir', type=str, default=".cache")
    parser.add_argument("--ndocs", type=int, default=10)
    # Attacker Arguments matching main.py
    parser.add_argument('--attack_method', type=str, default='LM_targeted')
    parser.add_argument('--adv_per_query', type=int, default=5, help='The number of adv texts for each target query.')
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
    parser.add_argument('--M', type=int, default=10, help='one of our parameters, the number of target queries')
    parser.add_argument('--eval_dataset', type=str, default="hotpotqa", help='BEIR dataset to evaluate') # Required for Attacker
    
    parser.add_argument("--world_size",  type=int, default=1)
    parser.add_argument("--dtype",  type=str, default="half")
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument("--use_groundness", action="store_true")
    parser.add_argument("--use_utility", action="store_true")
    parser.add_argument("--w_rel",  type=float, default=1.0)
    parser.add_argument("--w_sup",  type=float, default=1.0)
    parser.add_argument("--w_use",  type=float, default=1.0)
    parser.add_argument('--mode', type=str, default="adaptive_retrieval", choices=['adaptive_retrieval', 'no_retrieval', 'always_retrieve'])
    parser.add_argument('--metric', type=str, default="match")
    parser.add_argument('--ret_model_code', type=str, default="contriever", help="Retrieval model code for scoring") # New Arg
    
    args = parser.parse_args()

    # 1. Load Retrieval Model FIRST for Scoring
    # FIXME: check logic (Load Contriever explicitly for fair scoring)
    logger.info(f"Loading Retrieval Model {args.ret_model_code}...")
    ret_model, c_model, ret_tokenizer, get_emb_func = load_models(args.ret_model_code)
    
    ret_model.to(args.device)
    ret_model.eval()
    if c_model:
        c_model.to(args.device)
        c_model.eval()

    # Initialize Attacker
    # Attacker internal code uses relative path 'results/...' assuming root dir execution
    # We temporarily switch CWD to root_dir to satisfy this dependency
    cwd = os.getcwd()
    try:
        os.chdir(root_dir)
        attacker = Attacker(args,
                            model=ret_model,
                            c_model=c_model,
                            tokenizer=ret_tokenizer,
                            get_emb=get_emb_func)
    finally:
        os.chdir(cwd)

    # 2. Construct Dataset on-the-fly (Poisoned & Competed)
    # FIXME: check logic (Pass model to construction function)
    input_data = construct_hotpotqa_poisoned_data(
        args.queries_file,
        args.corpus_file,
        args.beir_results,
        args.adv_file,
        args.ndocs,
        ret_model,
        ret_tokenizer,
        args.device,
        attacker,
        args
    )
    
    del attacker
    if c_model: del c_model
    
    # 3. Preprocess
    input_data = preprocess_input_data(input_data, task=args.task)

    # 4. Load Gen Model (vLLM)
    # Note: vLLM loads on GPU, so manage memory if needed. 
    # But for 7B + Contriever it should be fine on decent GPU.
    # If OOM, one might need to delete ret_model after construction.
    del ret_model
    torch.cuda.empty_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    model = LLM(model=args.model_name, download_dir=args.download_dir,
                dtype=args.dtype, tensor_parallel_size=args.world_size)

    # 5. Load Tokens
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=args.use_groundness, use_utility=args.use_utility)

    # 6. Generate Loop
    def generate(prompt, evidences, max_new_tokens):
        return call_model_rerank_w_scores_batch(
            prompt, evidences=evidences, model=model, max_new_tokens=max_new_tokens,
            rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
            threshold=args.threshold, use_seqscore=args.use_seqscore,
            w_rel=args.w_rel, w_sup=args.w_sup, w_use=args.w_use, mode=args.mode, closed=False
        )

    preds = []
    prompts = []
    all_results = []
    asr_scores = [] # [METRIC] For ASR calculation # FIXME: check logic (New List for ASR)
    
    logger.info("Starting Poisoned Inference...")
    for i, row in tqdm(enumerate(input_data)):
        _, evidences = process_data_evidences(row, top_n=args.ndocs)
        prompt = row['instruction']
        
        pred, results, do_retrieve = generate(prompt, evidences, max_new_tokens=args.max_new_tokens)
        
        if type(pred) is str and (pred.startswith("#") or pred.startswith(":")):
            pred = pred[1:]
        
        # [METRIC] Compute Attack Success Rate (ASR)
        # FIXME: check logic (Added ASR Metric Calculation Logic)
        target_incorrect = row.get("target_incorrect_answer", "")
        # Use clean_str for robust comparison matching main.py
        is_success = 1 if target_incorrect and clean_str(target_incorrect) in clean_str(pred) else 0
        asr_scores.append(is_success)

        prompts.append(prompt)
        preds.append(pred)
        all_results.append(results)
        
        if i % 10 == 0:
            print(f"Current ASR: {np.mean(asr_scores):.4f}")
            final_results = {"preds": preds, "asr_mean": np.mean(asr_scores)}
            with open(args.output_file + "_tmp", "w") as outfile:
                json.dump(final_results, outfile)

    # Final Save
    final_results = {"preds": preds, "prompts": prompts, "all_results": all_results, "asr_mean": np.mean(asr_scores)}
    with open(args.output_file, "w") as outfile:
        json.dump(final_results, outfile)
    
    print("\n" + "="*30)
    print(f"Final Attack Success Rate (ASR): {np.mean(asr_scores):.4f}")
    print("="*30 + "\n")
    logger.info("Done.")

if __name__ == "__main__":
    main()
