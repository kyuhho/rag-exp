
import json
import os

# Paths
ADVERSARIAL_FILE = "results/adv_targeted_results/hotpotqa.json"
HOTPOT_DEV_FILE = "ReAct/data/hotpot_dev_v1_simplified.json"

def find_indices():
    with open(ADVERSARIAL_FILE, 'r') as f:
        adv_data = json.load(f)
    
    with open(HOTPOT_DEV_FILE, 'r') as f:
        dev_data = json.load(f)
    
    # Create a mapping from normalized question text to index in dev_data
    def normalize_q(q):
        return " ".join(q.strip().split())

    question_to_idx = {normalize_q(d['question']): i for i, d in enumerate(dev_data)}
    
    mapping = {}
    missing_questions = []
    found_count = 0
    for qid, entry in adv_data.items():
        q_text = normalize_q(entry['question'])
        if q_text in question_to_idx:
            mapping[qid] = question_to_idx[q_text]
            found_count += 1
        else:
            missing_questions.append(entry['question'])
            
    print(f"Found {found_count} out of {len(adv_data)} questions in dev_data.")
    if missing_questions:
        print("\nMissing Questions:")
        for q in missing_questions:
            print(f"- {q}")
    return mapping

if __name__ == "__main__":
    mapping = find_indices()
    with open("results/adv_targeted_results/qid_to_idx.json", "w") as f:
        json.dump(mapping, f)
