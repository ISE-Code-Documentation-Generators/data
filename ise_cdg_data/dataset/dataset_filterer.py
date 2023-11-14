import abc
from enum import Enum
import typing

import torch
from torchtext import vocab
from torch.utils.data import Dataset
from torch_geometric.data.data import Data as GeoData


class DatasetFilterByIndex(Dataset):

    def __init__(self, dataset: Dataset, indices: typing.List[int]) -> None:
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[self.indices[index]]