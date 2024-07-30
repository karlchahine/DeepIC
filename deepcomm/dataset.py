import numpy as np
import torch
import torch.utils.data as data


class RealData(data.Dataset):
    def __init__(self, path):
        """
        Argument: Path of torch saved .pt file
        saved as a tuple of (received LLRs, sent message)

        Usage eg:
        train_dset = RealData('data/train.pt')
        train_loader = data.DataLoader(train_dset, batch_size, shuffle=True) 

        """
        self.llrs, self.gts = torch.load(path)

    def __len__(self):
        return self.llrs.shape[0]

    def __getitem__(self, item):
        return self.llrs[item].float(), self.gts[item].float()
