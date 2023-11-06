from enum import Enum
from typing import Any, Union
from torch import Tensor, nn
from torch.nn.modules.module import Module
from torchtext import vocab

class EmbeddingAdapter:

    class VectorsType(Enum):
        GLOVE_6B = 'GloVe 6B'
        SIMPLE = 'simple'

    def __init__(self, vocab: vocab.Vocab) -> None:
        self.adaptee_vocab = vocab

    def __getattr__(self, name):
        return getattr(self.adaptee_vocab, name)
    
    @property
    def vocab_size(self):
        return len(self.adaptee_vocab)


    def get_embedding(self, embedding_size, vectors_type: 'EmbeddingAdapter.VectorsType') -> 'nn.Embedding':
        if vectors_type == self.VectorsType.GLOVE_6B:
            glove_vocab = vocab.GloVe(name='6B', dim=embedding_size)
            tokens_list = [self.get_itos(i) for i in range(self.vocab_size)]
            vecs = glove_vocab.get_vecs_by_tokens(tokens_list)
            glove_weights_subset = glove_vocab.vectors[vecs]
            return nn.Embedding.from_pretrained(glove_weights_subset, freeze=False)
        elif vectors_type == self.VectorsType.SIMPLE:
            return nn.Embedding(self.vocab_size, embedding_size)

