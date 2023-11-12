import abc
from enum import Enum
import typing

import torch
from torchtext import vocab
from torch.utils.data import Dataset
from torch_geometric.data.data import Data as GeoData



class Md4DefDatasetInterface(Dataset):

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        pass


class SourceGraphDatasetInterface(Dataset):
    @abc.abstractmethod
    def __getitem__(self, index) -> GeoData:
        pass



class Facade:

    class DatasetMode(Enum):
        WITH_PREPROCESS = 'with preprocess'
        WITHOUT_PREPROCESS = 'without preprocess'
    
    def __init__(self, mode: 'DatasetMode') -> None:
        self.mode = mode

    def get_md4def(self, path: str) -> 'Md4DefDatasetInterface':
        if self.mode == self.DatasetMode.WITH_PREPROCESS:
            from ise_cdg_data.dataset.markdown_source import Md4DefDatasetWithPreprocess
            return Md4DefDatasetWithPreprocess(path)
        else:
            from ise_cdg_data.dataset.markdown_source import Md4DefDatasetWithoutPreprocess
            return Md4DefDatasetWithoutPreprocess(path)
    
    def get_source_graph(self, path: str, src_vocab: vocab.Vocab) -> 'SourceGraphDatasetInterface':
        if self.mode == self.DatasetMode.WITH_PREPROCESS:
            from ise_cdg_data.dataset.source_graph import SourceGraphDatasetWithPreprocess
            return SourceGraphDatasetWithPreprocess(path, src_vocab)
        else:
            from ise_cdg_data.dataset.source_graph import SourceGraphDatasetWithoutPreprocess
            return SourceGraphDatasetWithoutPreprocess(path, src_vocab)
    
    def get_cnn2rnn(self, path: str, src_max_length: int) -> 'Md4DefDatasetInterface':
        if self.mode == self.DatasetMode.WITH_PREPROCESS:
            from ise_cdg_data.dataset.cnn2rnn import CNN2RNNDatasetWithPreprocess
            return CNN2RNNDatasetWithPreprocess(path, src_max_length)
        assert False, "mode is not supported"

    def get_cnn2rnn_features(self, path: str, src_max_length: int) -> 'Md4DefDatasetInterface':
        if self.mode == self.DatasetMode.WITH_PREPROCESS:
            from ise_cdg_data.dataset.cnn2rnn_features import CNN2RNNFeaturesDatasetWithPreprocess
            return CNN2RNNFeaturesDatasetWithPreprocess(path, src_max_length)
        assert False, "mode is not supported"