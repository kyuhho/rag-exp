
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

def main():
    corag_data = load_json('/home/work/Redteaming/rag-exp/results/trajectory_results/results_corag.json')
    poison_data = load_json('/home/work/Redteaming/rag-exp/results/adv_targeted_results/hotpotqa.json')
    
    poisoned_docs = []
    
    for item in corag_data:
        qid = item['id']
        subqueries = item.get('subqueries', [])
        
        if qid not in poison_data:
            continue
            
        entry = poison_data[qid]
        target_answer = entry['incorrect answer']
        adv_texts = entry['adv_texts']
        
        # 각 subquery마다 문서 생성
        for subquery_idx, subquery in enumerate(subqueries):
            # 각 subquery에 대해 모든 adv_texts를 사용하여 문서 생성
            for adv_idx, adv_text in enumerate(adv_texts):
                doc = {
                    "_id": f"poison_{qid}_subquery{subquery_idx}_adv{adv_idx}",
                    "title": subquery,  # subquery를 title로 사용
                    "text": adv_text,
                    "metadata": {
                        "original_id": qid,
                        "target_answer": target_answer,
                        "is_poisoned": True,
                        "subquery_index": subquery_idx
                    }
                }
                poisoned_docs.append(doc)
    
    print(f"Generated {len(poisoned_docs)} poisoned documents.")
    
    output_path = '/home/work/Redteaming/rag-exp/datasets/hotpotqa/poisoned_corpus_subquery.jsonl'
    save_jsonl(poisoned_docs, output_path)
    print(f"Poisoned corpus saved to {output_path}")

if __name__ == '__main__':
    main()