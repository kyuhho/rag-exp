
import os
import openai
import wikienv, wrappers
import json
import sys
import random
import time
import requests


from vllm import LLM, SamplingParams

# Initialize vLLM (limit memory to leave room for retriever index)
print("Initializing vLLM model...")
llm_model = LLM(
    model="meta-llama/Llama-2-7b-chat-hf", 
    dtype="half", 
    trust_remote_code=True,
    gpu_memory_utilization=0.6 # A6000(48GB) 기준, 인덱스(~11GB)를 고려하여 60%만 사용
)

def llm(prompt, stop=["\n"]):
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=100, stop=stop)
    outputs = llm_model.generate([prompt], sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text

# Environment Setup
import sys
# Add src to path to import E5_Retriever
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from e5_retriever import E5_Retriever
from e5_env import E5WikiEnv

# Initialize E5 Retriever
retriever = E5_Retriever(
    index_dir="datasets/hotpotqa/e5_index", 
    corpus_path="datasets/hotpotqa/corpus.jsonl",
    model_name="intfloat/e5-large-v2",
    device="cuda"
)

env = E5WikiEnv(retriever)
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

# ReAct Logic
folder = './prompts/'
# prompt_file = 'prompts_naive.json' # Adjusted path handling below in case running from different dir
prompt_file = os.path.join(os.path.dirname(__file__), 'prompts', 'prompts_naive.json')

with open(prompt_file, 'r') as f:
    prompt_dict = json.load(f)

webthink_examples = prompt_dict['webthink_simple6']
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples

def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        if to_print:
            print(step_str)
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    if to_print:
        print(info, '\n')
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info

# Main Loop
if __name__ == "__main__":
    idxs = list(range(7405))
    random.Random(233).shuffle(idxs)

    rs = []
    infos = []
    old_time = time.time()
    for i in idxs[:1]:
        r, info = webthink(i, to_print=True)
        rs.append(info['em'])
        infos.append(info)
        print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
        print('-----------')
        print()
