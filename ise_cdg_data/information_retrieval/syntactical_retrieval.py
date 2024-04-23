from rank_bm25 import BM25Okapi
from io import BytesIO
import pandas as pd


from ise_cdg_data.information_retrieval import CodeRetrieval
from ise_cdg_data.tokenize import get_source_and_markdown_tokenizers


class SyntacticalRetrival(CodeRetrieval):
    def __init__(self, model_name: str, code_dataset: list):
        super().__init__(model_name, code_dataset)
        self._code_tok = []
        self._model = None
        self._is_process = False
        self.tokenizer, _ = get_source_and_markdown_tokenizers()

    def process(self):
        self._code_tok = [self.tokenize(code) for code in self._code_dataset]
        self._model = BM25Okapi(self._code_tok)
        self._is_process = True

    def tokenize(self, code: str) -> list:
        return list(self.tokenizer(code))

    def set_emb(self, emb: list):
        self._code_tok = emb
        self._model = BM25Okapi(self._code_tok)
        self._is_process = True

    def get_similar(self, query: str) -> list:
        if not self._is_process:
            self.process()
        score = self._model.get_scores(self.tokenize(query)).tolist()
        score_dic = [{"corpus_id": i, "score": v} for i, v in enumerate(score)]
        sorted_score = sorted(score_dic, key=lambda x: x["score"], reverse=True)
        return sorted_score

    def get_dataframe(self):
        df = pd.DataFrame({"source": self._code_dataset, "embeding": self._code_tok})
        return df
