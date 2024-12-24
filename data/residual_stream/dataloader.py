import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from typing import Optional, Tuple

class ResidualStreamDataset(Dataset):
    def __init__(self, path: str, num_examples: Optional[int] = None):
        self.file = h5py.File(path, 'r')
        self.activations = self.file['activations']
        if num_examples is not None:
            self.activations = self.activations[:num_examples]
        
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.activations[idx]).view(torch.bfloat16).to(torch.float32) #load from uint16 to bfloat16, then cast to float32 to enable autocast. so yucky. there must be a better way
    
    def __del__(self):
        self.file.close()

def get_dataloader(
    path: str,
    batch_size: int,
    num_examples: Optional[int] = None,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    dataset = ResidualStreamDataset(path, num_examples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )