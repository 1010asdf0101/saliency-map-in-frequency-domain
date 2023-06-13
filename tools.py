from torch.utils.data import Dataset
import torchvision
from PIL import Image
import torch
import numpy as np
device = 'cpu'
h = 128
w = 128
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

class Train_DT(Dataset):
    def __init__(self, paths, is_gray=False):
        self.items=paths
        self.trans = torchvision.transforms.PILToTensor()
        self.resizer = torchvision.transforms.Resize((h, w), antialias=True)
        self.is_gray = is_gray
    def __len__(self):
        return len(self.items)
    def __getitem__(self, index):
        #print(self.items[index])
        img = Image.open(self.items[index])
        if self.is_gray: img = img.convert("L")
        else: img = img.convert("RGB")
        img = self.resizer(self.trans(img))
        return img
    
def collate_fn(batch):
    ret = torch.stack(batch).to(device)
    return ret

def gaussian_kernel(size, sigma):
    kernel = torch.Tensor(size, size)
    center = size // 2
    variance = sigma**2
    coefficient = 1.0 / (2 * np.pi * variance)
    for i in range(size):
        for j in range(size):
            distance = (i - center)**2 + (j - center)**2
            kernel[i, j] = coefficient * np.exp(-distance / (2 * variance))
    kernel = kernel / torch.sum(kernel)
    return kernel
