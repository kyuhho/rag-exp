
from wikienv import WikiEnv, textSpace

class E5WikiEnv(WikiEnv):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        # Re-initialize these as they might be reset in super().__init__ but we want to be sure
        self.observation_space = self.action_space = textSpace()

    def search_step(self, entity):
        # Override Wikipedia search with E5 Retrieve
        # We search for the top 1 document to simulate "going to a page"
        # The entity string is used as the query
        results = self.retriever.search(entity, k=1)

        print("="*30)
        print(f"[ENTITY] {entity}")
        print(f"[RESULT] {results}")
        print("="*30)
        
        if not results:
            self.obs = f"Could not find {entity}. Similar: []." 
            self.page = ""
        else:
            top_doc = results[0]
            title = top_doc.get('title', '')
            content = top_doc.get('contents', '')
            
            # Construct "Page" content simulating Wikipedia structure for lookup
            self.page = f"{title}\n{content}"
            self.obs = self.get_page_obs(self.page)
            self.lookup_keyword = self.lookup_list = self.lookup_cnt = None