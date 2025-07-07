import faiss, numpy as np

class VectorStore:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings))
        self.texts.extend(texts)

    def query(self, q_emb, k=3):
        D, I = self.index.search(np.array([q_emb]), k)
        return [self.texts[i] for i in I[0]]
