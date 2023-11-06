from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from ise_cdg_data import vocab

from ise_cdg_data.dataset import Facade
from ise_cdg_data.padding_collate import CNN2RNNCollate, PaddingCollate


def get_loader(
    dataset,
    src_vocab,
    md_vocab,
    batch_size,
    num_workers=0,
    shuffle=True,
    pin_memory=False,
):
    collate = PaddingCollate(
        src_vocab.get_stoi()["<pad>"],
        md_vocab.get_stoi()["<pad>"],
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    return loader


def geo_get_loader(
    dataset,
    batch_size,
    num_workers=0,
    shuffle=True,
    pin_memory=False,
):
    loader = GeoDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
    return loader


def get_cnn2rnn_loader(
    dataset,
    src_vocab,
    header_vocab,
    source_expected_sequence_length,
    batch_size,
    num_workers=0,
    shuffle=True,
    pin_memory=False,
):
    collate = CNN2RNNCollate(
        src_vocab.get_stoi()["<pad>"],
        header_vocab.get_stoi()["<pad>"],
        source_expected_sequence_length,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    return loader
