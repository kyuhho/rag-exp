
import json
import os

def prepare_poisoned_corpus(input_path, output_path):
    print(f"Loading adversarial data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    poisoned_docs = []
    for qid, entry in data.items():
        question = entry['question']
        target_answer = entry['incorrect answer']
        adv_texts = entry['adv_texts']
        
        for i, adv_text in enumerate(adv_texts):
            # Using _id and text to match original corpus.jsonl
            doc = {
                "_id": f"poison_{qid}_{i}",
                "title": question, # Use question as title for high retrieval score
                "text": adv_text,
                "metadata": {
                    "original_id": qid,
                    "target_answer": target_answer,
                    "is_poisoned": True
                }
            }
            poisoned_docs.append(doc)
            
    print(f"Generated {len(poisoned_docs)} poisoned documents.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in poisoned_docs:
            f.write(json.dumps(doc) + '\n')
    
    print(f"Poisoned corpus saved to {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "results/adv_targeted_results/hotpotqa.json"
    OUTPUT_FILE = "datasets/hotpotqa/poisoned_corpus.jsonl"
    prepare_poisoned_corpus(INPUT_FILE, OUTPUT_FILE)
