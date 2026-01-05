
from wikienv import WikiEnv, textSpace
import json
class E5WikiEnv(WikiEnv):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        # Re-initialize these as they might be reset in super().__init__ but we want to be sure
        self.observation_space = self.action_space = textSpace()
        self.last_search_poisoned = False
        self.any_poisoned_retrieved = False

    def _get_info(self):
        info = super()._get_info()
        info.update({
            "is_poisoned": self.last_search_poisoned,
            "any_poisoned": self.any_poisoned_retrieved
        })
        return info

    def reset(self, idx=None, seed=None, return_info=False, options=None):
        self.last_search_poisoned = False
        self.any_poisoned_retrieved = False
        return super().reset(seed=seed, return_info=return_info, options=options)

    def search_step(self, entity):
        # Override Wikipedia search with E5 Retrieve
        # We search for the top 1 document to simulate "going to a page"
        # The entity string is used as the query
        results = self.retriever.search(entity, k=1)
        
        if not results:
            self.obs = f"Could not find {entity}. Similar: []." 
            self.page = ""
            self.last_search_poisoned = False
        else:
            top_doc = results[0]
            title = top_doc.get('title', '')
            content = top_doc.get('contents', '')
            self.last_search_poisoned = top_doc.get('is_poisoned', False)
            if self.last_search_poisoned:
                self.any_poisoned_retrieved = True
            
            # Construct "Page" content simulating Wikipedia structure for lookup
            self.page = f"{title}\n{content}"
            self.obs = self.get_page_obs(self.page)
            self.lookup_keyword = self.lookup_list = self.lookup_cnt = None