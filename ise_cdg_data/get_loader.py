from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader

from ise_cdg_data.dataset import Facade
from ise_cdg_data.padding_collate import PaddingCollate


def get_loader(
    path,
    batch_size,
    num_workers=0,
    shuffle=True,
    pin_memory=False,
    is_dataset_processed=True,  
  ):
    mode = Facade.DatasetMode.WITHOUT_PREPROCESS if is_dataset_processed else Facade.DatasetMode.WITH_PREPROCESS

    dataset = Facade(mode).get_md4def(path)
    collate = PaddingCollate(
        dataset.src_vocab.get_stoi()['<pad>'],
        dataset.md_vocab.get_stoi()['<pad>'],
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    return loader, dataset

def geo_get_loader(
        path,
        src_vocab,
        batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=False,
        is_dataset_processed=True,
):
    mode = Facade.DatasetMode.WITHOUT_PREPROCESS if is_dataset_processed else Facade.DatasetMode.WITH_PREPROCESS
    dataset = Facade(mode).get_source_graph(path, src_vocab)
    loader = GeoDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
    return loader, dataset
