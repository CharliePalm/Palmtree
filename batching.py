from torch.utils.data import Dataset
import torch

class Data(Dataset):
    def __init__(self, x: list, device, y_c=None, y_f=None, y_t=None):
        if y_c is not None:
            self.x, self.y_c, self.y_f, self.y_t = torch.stack(x), torch.stack(y_c), torch.stack(y_f), torch.stack(y_t)
            self.y_c = self.y_c.to(dtype=torch.float32, device=device)
            self.y_f = self.y_f.to(dtype=torch.float32, device=device)
            self.y_t = self.y_t.to(dtype=torch.float32, device=device)
        else:
            self.x = torch.stack(x)
        self.x = self.x.to(dtype=torch.float32, device=device)
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y_c[idx], self.y_f[idx], self.y_t[idx]