from torch.nn.utils.rnn import pad_sequence

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
