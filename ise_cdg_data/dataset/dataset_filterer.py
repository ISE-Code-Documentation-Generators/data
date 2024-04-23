import typing

import torch
from torch.utils.data import Dataset

from ise_cdg_data.dataset import Md4DefDatasetInterface

class DatasetFilterByIndex(Dataset):

    def __init__(self, dataset: Dataset, indices: typing.List[int]) -> None:
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[self.indices[index]]
    

class DatasetFilterByIndexRaw(DatasetFilterByIndex):


    def __init__(self, dataset: Md4DefDatasetInterface, indices: typing.List[int]) -> None:
        super().__init__(dataset, indices)

    def get_raw_item(self, col, index):
        if isinstance(self.dataset, DatasetFilterByIndexRaw):
            return self.dataset.get_raw_item(col, self.indices[index])
        return self.dataset.df[col].iloc[self.indices[index]]