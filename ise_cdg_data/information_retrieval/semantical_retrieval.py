from sentence_transformers import SentenceTransformer, util
import pandas as pd
import ast
import torch

from ise_cdg_data.information_retrieval import CodeRetrieval


class SemanticalRetrieval(CodeRetrieval):
    _model_path = {
        "roberta": "flax-sentence-embeddings/st-codesearch-distilroberta-base",
        "mpnet": "all-mpnet-base-v2",
    }

    def __init__(self, model_name: str, code_dataset: list, device: torch.device):
        super().__init__(model_name, code_dataset)
        self._code_emb = [] * len(code_dataset)
        self._model = SentenceTransformer(self._model_path[model_name])
        self._model = self._model.to(device)
        self._is_process = False

    def process(self):
        self._code_emb = self._model.encode(self._code_dataset, convert_to_tensor=True)
        self._is_process = True

    def set_emb(self, emb: list):
        emb_numeric = [ast.literal_eval(e) for e in emb]
        self._code_emb = torch.tensor(emb_numeric)
        self._is_process = True

    def get_similar(self, query: str) -> list:
        if not self._is_process:
            self.process()
        query_emb = self._model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, self._code_emb)[0]
        return hits

    def get_dataframe(self):
        emb_list = [row_tensor.tolist() for row_tensor in self._code_emb]
        df = pd.DataFrame({"source": self._code_dataset, "embeding": emb_list})
        return df
