import abc
import typing
import ast

import torch
from torchtext import vocab
from torch.utils.data import Dataset
import pandas as pd

from data.tokenize import get_source_and_markdown_tokenizers
from data.dataset import Md4DefDatasetInterface


class Md4DefDataset(Md4DefDatasetInterface):
    df: pd.DataFrame

    def __init__(self, path):
        self.path = path
    
    @property
    def source(self) -> pd.Series:
        return self.df['source']

    @property
    def markdown(self) -> pd.Series:
        return self.df['markdown']

    def __len__(self):
        if not hasattr(self, 'df'):
            raise NotImplementedError('You must either fill the `df` field or override the `__len__` method')
        return len(self.df)

    @abc.abstractmethod
    def get_source_tensor(self, row: typing.Any) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_markdown_tensor(self, row: typing.Any) -> torch.Tensor:
        pass
    
    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        source = self.source.iloc[index]
        markdown = self.markdown.iloc[index]
        return self.get_source_tensor(source), self.get_markdown_tensor(markdown)


class Md4DefDatasetWithPreprocess(Md4DefDataset):

    @classmethod
    def vocab_factory(
        cls, tokenized_texts: typing.List[typing.Sequence[str]], min_freq=1
    ) -> vocab.Vocab:
        vocab_ = vocab.build_vocab_from_iterator(tokenized_texts, specials=[
            '<pad>', '<sos>', '<eos>', '<unk>'
        ], min_freq=min_freq)
        vocab_.set_default_index(vocab_.get_stoi()['<unk>'])
        return vocab_

    def __init__(
        self, path
    ):
        super().__init__(path)
        self.df = pd.read_csv(self.path, delimiter=',')


        self.src_tokenizer, self.md_tokenizer = get_source_and_markdown_tokenizers()
        self.src_vocab = self.vocab_factory(
            [self.src_tokenizer(src) for src in self.source],
            min_freq=3,
        )
        self.md_vocab = self.vocab_factory(
            [self.md_tokenizer(md) for md in self.markdown],
            min_freq=2,
        )

    def get_source_tensor(self, row: typing.Any) -> torch.Tensor:
        return torch.tensor([
            self.src_vocab.vocab.get_stoi()['<sos>'],
            *self.src_vocab(self.src_tokenizer(row)),
            self.src_vocab.vocab.get_stoi()['<eos>'],
        ])

    def get_markdown_tensor(self, row: typing.Any) -> torch.Tensor:
        return torch.tensor([
            self.src_vocab.vocab.get_stoi()['<sos>'],
            *self.md_vocab(self.md_tokenizer(row)),
            self.src_vocab.vocab.get_stoi()['<eos>'],
        ])


class Md4DefDatasetWithoutPreprocess(Md4DefDataset):

    def __init__(
        self, path
    ):
        super().__init__(path)
        df = pd.read_csv(self.path, delimiter=',', converters={
            'source': ast.literal_eval,
            'markdown': ast.literal_eval,
        })
        df = df[['source', 'markdown']]
        self.df = df
    
    def get_source_tensor(self, row: typing.Any) -> torch.Tensor:
        return torch.tensor(row)
    
    def get_markdown_tensor(self, row: typing.Any) -> torch.Tensor:
        return torch.tensor(row)
