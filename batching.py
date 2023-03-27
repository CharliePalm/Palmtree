from torch.utils.data import Dataset
import torch

class TrainData(Dataset):
    def __init__(self, x: list, y_c, y_f, y_t, device):
        self.x, self.y_c, self.y_f, self.y_t = torch.stack(x), torch.stack(y_c), torch.stack(y_f), torch.stack(y_t)
        self.x = self.x.to(dtype=torch.float32, device=device)
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y_c[idx], self.y_f[idx], self.y_t[idx]