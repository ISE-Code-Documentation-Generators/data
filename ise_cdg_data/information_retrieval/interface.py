from abc import ABC, abstractmethod
import torch

import pandas as pd



class CodeRetrieval(ABC):
    @abstractmethod
    def __init__(self, model_name: str, code_dataset: list):
        self._code_dataset = code_dataset
        self._model_name = model_name

    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def get_similar(self, query: str) -> list:
        pass

    @abstractmethod
    def set_emb(self, emb: list):
        pass



def create_semantical_ir_model(code_dataset: list, device: torch.device):
    from ise_cdg_data.information_retrieval.semantical_retrieval import SemanticalRetrieval
    return SemanticalRetrieval('roberta', code_dataset, device)


def load_semantical_ir_model(code_dataset: list, device: torch.device, embeding: list):
    from ise_cdg_data.information_retrieval.semantical_retrieval import SemanticalRetrieval
    model = SemanticalRetrieval('roberta', code_dataset, device)
    model.set_emb(embeding)
    return model


def load_code_metric_ir_model(code_dataset: list, embeding: list):
    from ise_cdg_data.information_retrieval.code_metric_retrieval import CodeMetricRetrieval
    model = CodeMetricRetrieval('code metric', code_dataset)
    model.set_emb(embeding)
    return model