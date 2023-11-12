import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F


class PaddingCollate:
    def __init__(self, input_pad_idx, target_pad_idx):
        self.input_pad_idx = input_pad_idx
        self.target_pad_idx = target_pad_idx

    def __call__(self, batch):
        inputs = [item[0] for item in batch]
        inputs = pad_sequence(
            inputs, batch_first=False, padding_value=self.input_pad_idx
        )
        targets = [item[1] for item in batch]
        targets = pad_sequence(
            targets, batch_first=False, padding_value=self.target_pad_idx
        )
        return inputs, targets


class CNN2RNNCollate:
    def __init__(
        self,
        source_pad_idx,
        header_pad_idx,
        source_expected_sequence_length,
    ):
        self.source_pad_idx = source_pad_idx
        self.header_pad_idx = header_pad_idx
        self.source_expected_sequence_length = source_expected_sequence_length

    def __call__(self, batch):
        # pad for data -> pad for cnn
        sources = [item[0] for item in batch]
        sources = self.__pad(
            self.source_pad_idx, sources
        )  # shape: (data_max_seq_length, batch)
        sources = self.__pad_to_length(
            sources, self.source_expected_sequence_length
        )  # shape: (max_seq_length, batch)

        # pad for data
        headers = [item[1] for item in batch]  # shape: (batch)
        headers = self.__pad(self.header_pad_idx, headers) # shape: (data_max_seq_length, batch)

        return (sources, headers)

    def __pad(self, pad_idx, texts):
        return pad_sequence(texts, batch_first=False, padding_value=pad_idx)

    @classmethod
    def __pad_to_length(cls, raw_input, expected_length):
        # raw_input.shape: (..., raw_input_length)
        raw_input_length = raw_input.shape[0]
        pad_length = expected_length - raw_input_length
        # shape: (..., expected_length)
        return F.pad(raw_input, (0, 0, 0, pad_length), value=0)

class CNN2RNNWithFeaturesCollate:
    def __init__(
        self,
        source_pad_idx,
        header_pad_idx,
        source_expected_sequence_length,
    ):
        self.source_pad_idx = source_pad_idx
        self.header_pad_idx = header_pad_idx
        self.source_expected_sequence_length = source_expected_sequence_length

    def __call__(self, batch):
        # pad for data -> pad for cnn
        sources = [item[0] for item in batch]
        sources = self.__pad(
            self.source_pad_idx, sources
        )  # shape: (data_max_seq_length, batch)
        sources = self.__pad_to_length(
            sources, self.source_expected_sequence_length
        )  # shape: (max_seq_length, batch)

        features = [item[1] for item in batch] # shape: (batch, features_length)
        features = torch.tensor(features)

        # pad for data
        headers = [item[2] for item in batch]  # shape: (batch)
        headers = self.__pad(self.header_pad_idx, headers) # shape: (data_max_seq_length, batch)

        return (sources, features, headers)

    def __pad(self, pad_idx, texts):
        return pad_sequence(texts, batch_first=False, padding_value=pad_idx)

    @classmethod
    def __pad_to_length(cls, raw_input, expected_length):
        # raw_input.shape: (..., raw_input_length)
        raw_input_length = raw_input.shape[0]
        pad_length = expected_length - raw_input_length
        # shape: (..., expected_length)
        return F.pad(raw_input, (0, 0, 0, pad_length), value=0)
