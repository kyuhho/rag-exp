import json
import os

# 파일 경로 설정
file_path = "/home/work/Redteaming/rag-exp/results/adv_targeted_results/hotpotqa.json"
save_path = "/home/work/Redteaming/rag-exp/results/adv_targeted_results/hotpotqa_x3.json"
def triple_all_adv_texts(path):
    # 파일이 존재하는지 확인
    if not os.path.exists(path):
        print(f"Error: 파일이 존재하지 않습니다. -> {path}")
        return

    # 1. JSON 파일 읽기
    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Error: JSON 파일 형식이 올바르지 않습니다.")
            return

    # 2. 모든 아이템을 순회하며 수정하기
    modified_count = 0
    
    # data가 딕셔너리 형태라고 가정 (ID가 key인 구조)
    for key, item in data.items():
        # item이 딕셔너리이고 'adv_texts' 키를 가지고 있는지 확인
        if isinstance(item, dict) and "adv_texts" in item:
            original_list = item["adv_texts"]
            
            # 리스트가 비어있지 않은 경우에만 작업 수행
            if original_list and isinstance(original_list, list):
                # 리스트를 3번 반복하여 확장 (예: 5개 -> 15개)
                new_list = original_list * 3
                item["adv_texts"] = new_list
                modified_count += 1

    print(f"총 {modified_count}개의 아이템이 수정되었습니다.")

    # 3. 변경된 내용을 파일에 다시 쓰기
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        print("파일 저장이 완료되었습니다.")

# 함수 실행
triple_all_adv_texts(file_path)