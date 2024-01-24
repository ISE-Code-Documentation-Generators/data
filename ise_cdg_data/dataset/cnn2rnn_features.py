import math
import typing
import markdown
import re

import torch
from torchtext import vocab
import pandas as pd
from ise_cdg_data.dataset.features_extractor import get_source_features_extractor

from ise_cdg_data.dataset.interface import Md4DefDatasetInterface
from ise_cdg_data.tokenize.interface import get_source_and_markdown_tokenizers


class CNN2RNNFeaturesDatasetWithPreprocess(Md4DefDatasetInterface):
    source_column = 'source'
    header_column = 'header'
    features_column = 'features'

    @classmethod
    def extract_headers(cls, markdown_text):
        try:
            # Parse the Markdown text using the markdown package
            html_text = markdown.markdown(markdown_text)
        except Exception as e:
            print('markdown text')
            print(markdown_text)
            raise e
        
        # Use regular expressions to extract headers from the HTML
        header_tags = re.findall(r'<h(\d)>(.*?)<\/h\1>', html_text)
        
        headers = []
        for tag, text in header_tags:
            level = int(tag)
            headers.append({'level': level, 'text': text})
        
        return headers

    @property
    def source(self) -> pd.Series:
        return self.df[self.source_column]

    @property
    def header(self) -> pd.Series:
        return self.df[self.header_column]
    
    @property
    def features(self) -> pd.Series:
        return self.df[self.features_column]

    @classmethod
    def vocab_factory(
        cls, tokenized_texts: typing.List[typing.Sequence[str]], min_freq=1
    ) -> 'vocab.Vocab':
        vocab_ = vocab.build_vocab_from_iterator(tokenized_texts, specials=[
            '<pad>', '<sos>', '<eos>', '<unk>'
        ], min_freq=min_freq)
        vocab_.set_default_index(vocab_.get_stoi()['<unk>'])
        return vocab_
        

    def __init__(self, path: str, src_max_length: int):
        super().__init__()
        self.path = path
        self.src_max_length = src_max_length

        df = pd.read_csv(self.path)
        print(df.columns)
        df[self.features_column] = get_source_features_extractor().extract(df['source'])
        df = self.add_header_column(df)
        df = df[[self.source_column, self.header_column, self.features_column]]
        self.df = df
        self.filter_df()

        self.src_tokenizer, self.md_tokenizer = get_source_and_markdown_tokenizers(cleanse_markdown=False)
        self.src_vocab = self.vocab_factory(
            [self.src_tokenizer(src) for src in self.source],
            min_freq=3, 
        )
        self.md_vocab = self.vocab_factory(
            [self.md_tokenizer(md) for md in self.header],
        )


    def add_header_column(self, df):
        df = df[df['markdown'].apply(type) == str] # null markdown exists :)
        markdown_headers = df['markdown'].apply(self.extract_headers)
        markdown_headers = markdown_headers.apply(lambda headers: list(map(lambda header: header['text'], headers)))
        df = df[markdown_headers.apply(lambda headers: len(headers) != 0)]
        df = df.assign(header=markdown_headers).explode('header')
        return df
    
    def filter_source(self, tokenizer):
      tokenized_rows = self.df[self.source_column].apply(tokenizer).apply(len)
      self.df = self.df[tokenized_rows <= self.src_max_length]

    def filter_header_max_length(self, tokenizer):
      tokenized_rows: 'pd.Series' = self.df[self.header_column].apply(tokenizer).apply(len)
      max_length_tokenized_rows = tokenized_rows.sort_values()
      max_length = max_length_tokenized_rows.iloc[math.floor(len(self.df) *  0.95)]
      self.df = self.df[tokenized_rows <= max_length]

    def filter_header_min_length(self, tokenizer):
      min_length = 3
      tokenized_rows = self.df[self.header_column].apply(tokenizer).apply(len)
      self.df = self.df[tokenized_rows >= min_length]

    def filter_df(self):
        src_tokenizer, md_tokenizer = get_source_and_markdown_tokenizers(cleanse_markdown=False)
        self.filter_source(src_tokenizer)
        self.filter_header_max_length(md_tokenizer)
        self.filter_header_min_length(md_tokenizer)

    def get_source_tensor(self, row: typing.Any) -> torch.Tensor:
        return torch.tensor([
            self.src_vocab.vocab.get_stoi()['<sos>'],
            *self.src_vocab(self.src_tokenizer(row)),
            self.src_vocab.vocab.get_stoi()['<eos>'],
        ])
    
    def get_features_tensor(self, row: typing.Any) -> torch.Tensor:
        return torch.tensor(row)

    def get_header_tensor(self, row: typing.Any) -> torch.Tensor:
        return torch.tensor([
            self.src_vocab.vocab.get_stoi()['<sos>'],
            *self.md_vocab(self.md_tokenizer(row)),
            self.src_vocab.vocab.get_stoi()['<eos>'],
        ])

    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        header = self.header.iloc[index]
        features = self.features.iloc[index]
        source = self.source.iloc[index]
        return self.get_source_tensor(source), self.get_features_tensor(features), self.get_header_tensor(header)
    
    def __len__(self) -> int:
        return len(self.df)
