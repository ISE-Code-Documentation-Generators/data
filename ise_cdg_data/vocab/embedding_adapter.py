from enum import Enum
from typing import Any, Union
from torch import Tensor
from torch.nn.modules.module import Module
from torchtext import vocab

class EmbeddingAdapter:

    class VectorsType(Enum):
        GLOVE_6B = 'GloVe 6B'

    def __init__(self, vocab: vocab.Vocab) -> None:
        self.adaptee_vocab = vocab

    def __getattr__(self, name):
        return getattr(self.adaptee_vocab, name)
    
    def get_embedding(self, embedding_size, vectors_type: 'EmbeddingAdapter.VectorsType'):
        if vectors_type == self.VectorsType.GLOVE_6B:
            vocab.GloVe(name='6B', dim=embedding_size)