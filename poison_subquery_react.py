
import json
import os

def load_jsonl(path: str):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def save_jsonl(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def extract_search_keyword(action: str) -> str:
    """Extract keyword from Search[keyword] action."""
    if not action.lower().startswith('search['):
        return None
    
    # Extract content between [ and ]
    start_idx = action.find('[')
    end_idx = action.find(']')
    
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        return None
    
    keyword = action[start_idx + 1:end_idx].strip()
    return keyword if keyword else None

def main():
    react_data = load_json('/home/work/Redteaming/rag-exp/results/trajectory_results/poisoned_results_react.json')
    poison_data = load_json('/home/work/Redteaming/rag-exp/results/adv_targeted_results/hotpotqa.json')
    
    poisoned_docs = []
    
    for item in react_data:
        qid = item['id']
        steps = item.get('steps', [])
        
        if qid not in poison_data:
            continue
            
        entry = poison_data[qid]
        target_answer = entry['incorrect answer']
        adv_texts = entry['adv_texts']
        
        # 각 step에서 Search action만 처리
        for step_idx, step in enumerate(steps):
            action = step.get('action', '')
            keyword = extract_search_keyword(action)
            
            # Search action이 아닌 경우 스킵
            if keyword is None:
                continue
            
            # 각 Search action에 대해 모든 adv_texts를 사용하여 문서 생성
            for adv_idx, adv_text in enumerate(adv_texts):
                doc = {
                    "_id": f"poison_{qid}_step{step_idx}_adv{adv_idx}",
                    "title": keyword,  # Search action의 keyword를 title로 사용
                    "text": adv_text,
                    "metadata": {
                        "original_id": qid,
                        "target_answer": target_answer,
                        "is_poisoned": True,
                        "step_index": step_idx,
                        "action": action
                    }
                }
                poisoned_docs.append(doc)

    print(f"Generated {len(poisoned_docs)} poisoned documents.")
    output_path = '/home/work/Redteaming/rag-exp/datasets/hotpotqa/poisoned_corpus_subquery_react.jsonl'
    save_jsonl(poisoned_docs, output_path)
    print(f"Poisoned corpus saved to {output_path}")

if __name__ == '__main__':
    main()