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
        
        # Convert query to a tensor and move it to the device
        query_tensor = torch.tensor(query, device=self._device).float()
        
        # Normalize the query tensor
        query_tensor = query_tensor / query_tensor.norm(dim=0, keepdim=True)
        
        # Normalize the _code_emb tensor along the second dimension
        Y = self._code_emb
        Y = Y / Y.norm(dim=1, keepdim=True)
        
        # Compute cosine similarity using matrix multiplication
        scores = torch.matmul(Y, query_tensor)
        
        # Get sorted indices based on the scores
        sorted_indices = torch.argsort(scores, descending=True)
        
        # Gather sorted scores
        sorted_scores = scores[sorted_indices]
        
        # Prepare the result as a list of dictionaries
        result = [
            {"corpus_id": sorted_indices[idx].item(), "score": sorted_scores[idx].item()}
            for idx in range(len(sorted_scores))
        ]
        
        return result


    def get_dataframe(self):
        assert self._code_emb is not None, "use `set_emb` first"
        emb_list = [row_tensor.tolist() for row_tensor in self._code_emb]
        df = pd.DataFrame({"source": self._code_dataset, "embeding": emb_list})
        return df
