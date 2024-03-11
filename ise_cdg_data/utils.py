import math
import os
import random
import typing
import torch
from torch.utils.data import Dataset
from ise_cdg_data.dataset.markdown_source import Md4DefDataset
import pandas as pd


class LazyFileName:
    def __init__(self, path_to_file) -> None:
        self.__path = path_to_file
        self.__base_name, self.__name, self.__extension = [None] * 3

    def __set_fields(self):
        if self.__base_name is not None:  # Guard
            return
        self.__base_name = os.path.basename(self.__path)
        self.__name, self.__extension = os.path.splitext(self.__base_name)

    @property
    def base_name(self):
        self.__set_fields()
        return self.__base_name

    @property
    def name(self):
        self.__set_fields()
        return self.__name

    @property
    def extension(self):
        self.__set_fields()
        return self.__extension


def get_range(batch_size, current_batch, size):
    return range(
        current_batch * batch_size, min(size, (current_batch + 1) * batch_size)
    )


def train_test_split(size: int, test_portion: float):
    test_size = int(math.ceil(test_portion * size))
    test_indices = random.sample(range(0, size), test_size)
    train_indices = [ind for ind in range(0, size) if ind not in test_indices]
    return train_indices, test_indices


class Md4DefDatasetFilterDecorator(Dataset):

    def __init__(self, dataset: Md4DefDataset):
        self.dataset = dataset
        self._saved_df = None

    def get_current_dataframe(self) -> pd.DataFrame:
        return self.dataset.df

    def set_filtered_dataframe(self, dataframe: pd.DataFrame):
        if self._saved_df is None:
            self._saved_df = self.get_current_dataframe()
        self.dataset.df = dataframe
    
    def reset_dataframe(self):
        self.dataset.df = self._saved_df or self.dataset.df
        self._saved_df = None

    def __len__(self) -> int:
        return len(self.dataset)
        
    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[index]