from typing import List
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import ast
import torch

from ise_cdg_data.information_retrieval import CodeRetrieval


class CodeMetricRetrieval(CodeRetrieval):

    def __init__(self, model_name: str, code_dataset: list, device: torch.device):
        super().__init__(model_name, code_dataset)
        self._code_emb = None  
        self._device = device
    
    def process(self):
        raise Exception("Not possible to process. use `set_emb`")

    def set_emb(self, emb: list):
        self._code_emb = torch.tensor(emb)
        self._code_emb = self._code_emb.to(self._device)
        self._is_process = True

    def get_similar(self, query: list) -> list:
        if not self._is_process:
            self.process()
        query_tensor = torch.tensor(query)
        query_tensor = query_tensor.to(self._device)

        def cosine_similarity(x, y):
            return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))
        
        x = query_tensor
        Y = self._code_emb

        # Calculate cosine similarity scores
        scores = [cosine_similarity(x, y_i) for y_i in Y]

        # Get sorted indices based on the scores
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Sort scores accordingly
        sorted_scores = [scores[i] for i in sorted_indices]

        result = [{'corpus_id': idx, 'score': score} for idx, score in sorted_scores]


        return result

    def get_dataframe(self):
        assert self._code_emb is not None, "use `set_emb` first"
        emb_list = [row_tensor.tolist() for row_tensor in self._code_emb]
        df = pd.DataFrame({"source": self._code_dataset, "embeding": emb_list})
        return df
