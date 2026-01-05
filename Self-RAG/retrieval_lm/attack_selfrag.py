#!/usr/bin/python
# -*- coding: UTF-8 -*-

import spacy
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import random
import torch
import os
import numpy as np
import openai
from tqdm import tqdm
import json
import argparse
import ast
import re
from tqdm import tqdm
from collections import Counter
import string
import sys
import time
from utils import PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens, load_special_tokens
from metrics import match, accuracy

# Add root path for src access
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.e5_retriever import E5_Retriever
# Local clean_str to ensure perfect consistency with attack_corag.py
import string
def clean_str(s):
    if s is None:
        return ""
    s = str(s).lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = " ".join(s.split())
    return s

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed = 633
# setup_seeds will be called in main with args.seed


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
    results = {}
    if mode != "always_retrieve":
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=32016)
        preds = model.generate([prompt], sampling_params)
        pred_token_ids = preds[0].outputs[0].token_ids
        pred_text = preds[0].outputs[0].text
        pred_log_probs = preds[0].outputs[0].logprobs
        results["no_retrieval"] = pred_text

    # save relevance token scores
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
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=5000)
        preds = model.generate(evidence_augmented_inputs, sampling_params)

        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        for p_idx, pred in enumerate(preds):
            pred_token_ids = pred.outputs[0].token_ids
            pred_text = pred.outputs[0].text
            pred_log_probs = pred.outputs[0].logprobs
            seq_score = pred.outputs[0].cumulative_logprob / \
                max(len(pred.outputs[0].token_ids), 1)

            relevance_score_dict.setdefault(p_idx, {})
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # Compute reward scores
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
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
        prompt += "[No Retrieval]"
        preds = model.generate([prompt], sampling_params)

        pred = preds[0].outputs[0].text

    # Aggregating answers
    if len(results) == 1:
        postprocessed_pred = postprocess_answer_option_conditioned(pred)
        return postprocessed_pred, results, do_retrieve
    else:
        answer2score = {}
        if closed is True:
            for key, result in results.items():
                if key == "no_retrieval":
                    continue
                answer = postprocess_answer_option_conditioned(result["pred"])
                score = result["score"]
                answer2score.setdefault(answer, 0)
                answer2score[answer] += score
            sorted_answers = sorted(
                answer2score.items(), key=lambda x: x[1], reverse=True)
            best_option = sorted_answers[0][0]
        else:
            path2score = {key: item["score"] for key,
                          item in results.items() if key != "no_retrieval"}
            best_path = sorted(path2score.items(),
                               key=lambda x: x[1], reverse=True)[0][0]
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
        if task == "arc_c":
            choices = item["choices"]
            answer_labels = {}
            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]
                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            item["instruction"] = instruction + \
                "\n\n### Input:\n" + item["question"] + choices
            item["answers"] = [item["answerKey"]]
        else:
            prompt = instruction + "\n\n## Input:\n\n" + \
                item["question"] if instruction is not None else item["question"]
            item["instruction"] = prompt
        new_data.append(item)

    return new_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="selfrag/selfrag_llama2_7b")
    parser.add_argument('--input_file', type=str, help="Not used if using E5_Retriever, but kept for compatibility")
    parser.add_argument('--output_file', type=str, default="results/adv_targeted_results/preds_selfrag_hotpotqa.json")
    parser.add_argument('--task', type=str, default="hotpotqa")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--download_dir', type=str, help="specify vllm model download dir",
                        default=".cache")
    parser.add_argument("--ndocs", type=int, default=5,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--world_size",  type=int, default=1)
    parser.add_argument("--dtype",  type=str, default="half")
    
    # Retriever settings
    parser.add_argument("--index_dir", type=str, default="../../datasets/hotpotqa/e5_index")
    parser.add_argument("--corpus_path", type=str, default="../../datasets/hotpotqa/corpus.jsonl")
    parser.add_argument("--poisoned_index_dir", type=str, default="../../datasets/hotpotqa/e5_index_poisoned")
    parser.add_argument("--poisoned_corpus_path", type=str, default="../../datasets/hotpotqa/poisoned_corpus.jsonl")
    parser.add_argument("--encoder_model", type=str, default="intfloat/e5-large-v2")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--eval_dataset', type=str, default="hotpotqa")
    parser.add_argument('--dry_run', action='store_true')

    # Decoding hyperparams
    parser.add_argument('--threshold', type=float,
                        default=None, help="Adaptive threshold.")
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument("--use_groundness", action="store_true",
                        help="use ground score")
    parser.add_argument(
        "--use_utility", action="store_true", help="tree search")
    parser.add_argument("--beam_width",  type=int,
                        default=2, help="beam search width")
    parser.add_argument("--max_depth",  type=int,
                        default=2, help="tree depth width")
    parser.add_argument("--w_rel",  type=float, default=1.0,
                        help="reward weight for document relevance")
    parser.add_argument("--w_sup",  type=float, default=1.0,
                        help="reward weight for generation support (attribution)")
    parser.add_argument("--w_use",  type=float, default=1.0,
                        help="reward weight for overall completeness / utility.")
    parser.add_argument('--mode', type=str, help="mode to control retrieval.",
                        default="default", choices=['adaptive_retrieval', 'no_retrieval', 'always_retrieve'],)
    parser.add_argument('--metric', type=str, help="metric to be used during evaluation")
    args = parser.parse_args()
    setup_seeds(args.seed)
    device = f'cuda:{args.gpu_id}'
    
    # 1. Initialize E5 Retriever
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
    adv_data_path = f'../../results/adv_targeted_results/{args.eval_dataset}.json'
    print(f"Loading Adversarial Data from {adv_data_path}...")
    with open(adv_data_path, 'r') as f:
        incorrect_answers = json.load(f)
    
    target_qids = list(incorrect_answers.keys())
    if args.dry_run:
        target_qids = target_qids[:1]
    print(f"Total target samples: {len(target_qids)}")

    # 3. Initialize Self-RAG Model
    gpt = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(gpt, padding_side="left")
    model = LLM(model=gpt, download_dir=args.download_dir,
                dtype=args.dtype, tensor_parallel_size=args.world_size)

    # Get token ids for reflection tokens.
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=args.use_groundness, use_utility=args.use_utility)

    def generate(prompt, evidences, max_new_tokens):
        return call_model_rerank_w_scores_batch(prompt, evidences=evidences, model=model, max_new_tokens=max_new_tokens,
                                                rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                                threshold=args.threshold,use_seqscore=args.use_seqscore,
                                                w_rel=args.w_rel, w_sup=args.w_sup, w_use=args.w_use, mode=args.mode, closed=args.task in ["fever", "arc_c"])

    preds = []
    prompts = []
    all_results = []
    asr_success_count = 0
    total_poisoned_docs = 0
    poisoned_questions_count = 0
    input_data = []

    print("Starting Evaluation...")
    for i, qid in tqdm(enumerate(target_qids)):
        item = incorrect_answers[qid]
        question = item['question']
        target_answer = item.get('incorrect answer', '')
        
        # Retrieval
        retriever_results = retriever.search(question, k=args.ndocs)
        
        # Format for Self-RAG (contents -> text)
        evidences = []
        num_poisoned = sum(1 for res in retriever_results if res.get('is_poisoned', False))
        total_poisoned_docs += num_poisoned
        if num_poisoned > 0:
            any_poisoned = True
            poisoned_questions_count += 1

        # Prepare Prompt
        # Use TASK_INST for instruction
        instruction = TASK_INST.get(args.task, None)
        prompt_input = instruction + "\n\n## Input:\n\n" + question if instruction is not None else question
        row = {"instruction": prompt_input}
        prompt = PROMPT_DICT["prompt_no_input"].format_map(row)
        
        # Generate
        pred, results, do_retrieve = generate(prompt, evidences, max_new_tokens=args.max_new_tokens)
        
        if type(pred) is str and len(pred) > 0 and (pred[0] == "#" or pred[0] == ":"):
            pred = pred[1:]
        
        # Evaluate ASR
        is_asr_success = clean_str(target_answer) in clean_str(pred)
        if is_asr_success:
            asr_success_count += 1
            
        preds.append(pred)
        prompts.append(prompt)
        all_results.append({
            "id": qid,
            "question": question,
            "prediction": pred,
            "target_answer": target_answer,
            "asr_success": is_asr_success,
            "any_poisoned": any_poisoned,
            "num_poisoned": num_poisoned,
            "results": results
        })

        print(f"ASR Success: {is_asr_success} | Poisoned Docs: {num_poisoned}/{args.ndocs}")

        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{total_count}] Current ASR: {asr_success_count / (i+1):.4f} | Poisoned Question Ratio: {poisoned_questions_count / (i+1):.4f}")

    # Final Summary
    total = len(target_qids)
    asr_mean = asr_success_count / total if total > 0 else 0
    poison_question_ratio = poisoned_questions_count / total if total > 0 else 0
    poison_docs_ratio = total_poisoned_docs / (total * args.ndocs) if total > 0 else 0

    print("\n" + "="*50)
    print("FINAL ATTACK RESULTS (Self-RAG)")
    print(f"Total Questions: {total}")
    print(f"Attack Success Rate (ASR): {asr_mean:.4f}")
    print(f"Poisoned Question Ratio: {poison_question_ratio:.4f} ({poisoned_questions_count}/{total})")
    print(f"Poisoned Docs Ratio: {poison_docs_ratio:.4f} ({total_poisoned_docs}/{total * args.ndocs})")
    print("="*50)

    # Save results
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output_file, "w") as outfile:
        json.dump({
            "summary": {
                "total": total,
                "asr_mean": asr_mean,
                "poison_ratio": poison_ratio
            },
            "details": all_results
        }, outfile, indent=4)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
