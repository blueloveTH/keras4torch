from torch.utils.data import Dataset, Sampler
import torch

class SlicedDataset(Dataset):
    """
    Create a sliced dataset. It just keeps a reference of `array` thus can avoid memory copy.

    Args:

    * `slice` (1D-array or list): The slice sequence.

    * `array` (`ndarray` or `torch.Tensor`): Your data.
    """
    def __init__(self, slice, *array):
        super(SlicedDataset, self).__init__()
        self.array = array
        self.slice = slice

    def __len__(self):
        return len(self.slice)

    def __getitem__(self, index):
        index = self.slice[index]
        return [a[index] for a in self.array]

class RestrictedRandomSampler(Sampler):
    def __init__(self, cnt_list: list) -> None:
        self.cnt_list = cnt_list.copy()

    def __iter__(self):
        for cnt in self.cnt_list:
            yield from torch.randperm(cnt)

    def __len__(self):
        return sum(self.cnt_list)

__all__ = ['SlicedDataset', 'RestrictedRandomSampler']