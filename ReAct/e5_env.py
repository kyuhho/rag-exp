
from wikienv import WikiEnv, textSpace
import json
class E5WikiEnv(WikiEnv):
    def __init__(self, retriever, k=1):
        super().__init__()
        self.retriever = retriever
        self.k = k
        # Re-initialize these as they might be reset in super().__init__ but we want to be sure
        self.observation_space = self.action_space = textSpace()
        self.last_search_poisoned = False
        self.any_poisoned_retrieved = False
        self.last_search_poisoned_count = 0
        self.last_search_total_count = 0
        self.last_search_poisoned_flags = []  # 각 문서별 poisoned 여부 리스트

    def _get_info(self):
        info = super()._get_info()
        info.update({
            "is_poisoned": self.last_search_poisoned,
            "any_poisoned": self.any_poisoned_retrieved,
            "poisoned_count": self.last_search_poisoned_count,
            "total_count": self.last_search_total_count,
            "poisoned_flags": self.last_search_poisoned_flags  # 각 문서별 poisoned 여부 리스트
        })
        return info

    def reset(self, idx=None, seed=None, return_info=False, options=None):
        self.last_search_poisoned = False
        self.any_poisoned_retrieved = False
        self.last_search_poisoned_count = 0
        self.last_search_total_count = 0
        self.last_search_poisoned_flags = []
        return super().reset(seed=seed, return_info=return_info, options=options)

    def search_step(self, entity):
        # Override Wikipedia search with E5 Retrieve
        # We search for the top k documents
        results = self.retriever.search(entity, k=self.k)
        poisoned_cnt = sum(1 for res in results if res.get('is_poisoned', False))
        total_cnt = len(results)
        print(f"[Retriever Results] {results}")
        print(f"Poisoned Count: {poisoned_cnt}/{total_cnt}")
        print("-"*50)

        
        if not results:
            self.obs = f"Could not find {entity}. Similar: []." 
            self.page = ""
            self.last_search_poisoned = False
            self.last_search_poisoned_count = 0
            self.last_search_total_count = 0
            self.last_search_poisoned_flags = []
        else:
            self.page = ""
            obs_list = []
            self.last_search_poisoned = False
            self.last_search_poisoned_count = poisoned_cnt
            self.last_search_total_count = total_cnt
            self.last_search_poisoned_flags = []  # 초기화
            
            for res in results:
                title = res.get('title', '')
                content = res.get('contents', '')
                is_poisoned = res.get('is_poisoned', False)
                
                # Update poisoned flags
                if is_poisoned:
                    self.last_search_poisoned = True
                    self.any_poisoned_retrieved = True
                
                # 각 문서의 poisoned 여부를 리스트에 저장
                self.last_search_poisoned_flags.append(is_poisoned)
                
                # 1. Full page content for lookup (simulating Wikipedia structure)
                if self.page:
                    self.page += "\n\n"
                self.page += f"{title}\n{content}"
                
                # 2. Get observation (first 5 sentences) for this document
                doc_obs = self.get_page_obs(f"{title}\n{content}")
                obs_list.append(doc_obs)
            
            # Combine observations
            self.obs = "\n".join(obs_list) if self.k > 1 else obs_list[0]
            self.lookup_keyword = self.lookup_list = self.lookup_cnt = None