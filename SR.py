import torch
import math
import numpy as np
import torch, torchvision
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import cv2, time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
device = torch.device('cuda:3')
paths = Path('/home/shawnman99/data/cream/Coining/damaged_bump').rglob('*.png')

## build dataloader  &  define required functions
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

class Train_DT(Dataset):
    def __init__(self, paths):
        trans = torchvision.transforms.PILToTensor()
        self.items=[]
        for p in paths:
            img = Image.open(str(p)).convert("L")
            self.items.append(trans(img))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, index):
        return self.items[index]
    
def collate_fn(batch):
    ret = torch.stack(batch).to(device)
    return ret

def gaussian_kernel(size, sigma):
    kernel = torch.Tensor(size, size)
    center = size // 2
    variance = sigma**2
    coefficient = 1.0 / (2 * math.pi * variance)
    for i in range(size):
        for j in range(size):
            distance = (i - center)**2 + (j - center)**2
            kernel[i, j] = coefficient * math.exp(-distance / (2 * variance))
    kernel = kernel / torch.sum(kernel)
    return kernel

def main():
    BATCH_SIZE = 16
    h, w = 64, 64
    dataset = Train_DT(paths)
    train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    resizer = torchvision.transforms.Resize((h, w), antialias=True)
    kernel = torch.ones((1, 1, w, h), dtype=torch.float32) / h / w
    kernel = kernel.to(device)
    gk = gaussian_kernel(5, 8)
    gk = gk.reshape((1, 1, 5, 5)).to(device)
    start = time.time()
    for b in train_loader:
        b = resizer(b)
        fft = torch.fft.fft2(b)
        logAmplitude = torch.log(torch.abs(fft))
        Phase = torch.angle(fft)
        avgLogAmp = torch.matmul(logAmplitude, kernel)
        spectralResidual = logAmplitude - avgLogAmp
        salencyMap = torch.fft.ifft2(torch.exp(spectralResidual+1j*Phase))
        salencyMap = salencyMap.abs()**2
        salencyMap = torch.nn.functional.conv2d(salencyMap, gk, padding=2, stride=1)
        salMap = salencyMap[0].squeeze().detach().cpu().numpy()
        print(salMap.max(), salMap.min())
        # Display the Result.
        img = b[0].permute([1, 2, 0]).detach().cpu().numpy()
        plt.subplot(1, 2, 1), plt.imshow(img)
        plt.subplot(1, 2, 2), plt.imshow(salMap)
        plt.show()
    end = time.time()
    print(f"time : {end-start} / per image : {(end-start)/len(dataset)}")
if __name__ == '__main__':
    main()