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
paths = Path('/home/shawnman99/data/fst_pellicle_sam_labeled/train').rglob('*.png')

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
            w, h = img.size
            cropped = img.crop((0, 0, w, h))
            self.items.append(trans(cropped))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, index):
        return self.items[index]
    
def collate_fn(batch):
    ret = torch.stack(batch).to(device)
    return ret

import torch
import math

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
    dataset = Train_DT(paths)
    train_loader = DataLoader(dataset, 1024, shuffle=False, collate_fn=collate_fn)
    resizer = torchvision.transforms.Resize((256, 256), antialias=True)
    kernel = torch.ones((1, 1, 3, 3)) / 9.
    kernel = kernel.to(device)
    gk = gaussian_kernel(5, 8)
    gk = gk.reshape((1, 1, 5, 5)).to(device)
    start = time.time()
    for b in train_loader:
        b = resizer(b)
        fft = torch.fft.fft2(b)
        fft+=1
        logAmplitude = torch.log(torch.abs(torch.fft.fftshift(fft)))
        Phase = torch.angle(fft)
        avgLogAmp = torch.nn.functional.conv2d(logAmplitude, kernel, padding=1)
        spectralResidual = logAmplitude - avgLogAmp
        salencyMap = torch.abs(torch.fft.ifft2(torch.exp(spectralResidual+1j*Phase)))**2
        print(salencyMap.shape, gk.shape)
        salencyMap = torch.nn.functional.conv2d(salencyMap, gk, padding=2)
        salMap = salencyMap[0].squeeze().detach().cpu().numpy()
        print(salMap.max(), salMap.min())
        # Display the Result.
        plt.imshow(salMap)
        plt.show()
    end = time.time()
    print(f"time : {end-start} / per image : {(end-start)/len(dataset)}")
if __name__ == '__main__':
    main()