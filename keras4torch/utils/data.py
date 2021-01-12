from torch.utils.data import Dataset

class SlicedDataset(Dataset):
    def __init__(self, slice, *array):
        super(SlicedDataset, self).__init__()
        self.array = array
        self.slice = slice

    def __len__(self):
        return len(self.slice)

    def __getitem__(self, index):
        index = self.slice[index]
        return [a[index] for a in self.array]

__all__ = ['SlicedDataset']