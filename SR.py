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
#paths = Path('/home/shawnman99/data/ss_aoi/BF40077_211008_042659_mv').rglob('*.jpg')
paths = Path('/home/shawnman99/img_intern/sin3').rglob('*.bmp')

## build dataloader  &  define required functions
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

h, w = 128, 128

class Train_DT(Dataset):
    def __init__(self, paths):
        trans = torchvision.transforms.PILToTensor()
        resizer = torchvision.transforms.Resize((h, w), antialias=True)
        self.items=[]
        for p in paths:
            img = Image.open(str(p)).convert("L")
            img = resizer(trans(img))
            self.items.append(img)
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
    BATCH_SIZE = 256
    dataset = Train_DT(paths)
    train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32) / 9.
    kernel = kernel.to(device)
    gk = gaussian_kernel(5, 8)
    gk = gk.reshape((1, 1, 5, 5)).to(device)
    start = time.time()
    for b in train_loader:
        fft = torch.fft.fft2(b)
        mag = torch.abs(fft)
        La = torch.log(mag)
        padded_La = torch.nn.ReflectionPad2d(1)(La)
        avgLogAmp = torch.nn.functional.conv2d(padded_La, kernel)
        SR = torch.exp(La - avgLogAmp)
        fft.real = fft.real * SR / mag
        fft.imag = fft.imag * SR / mag
        f = torch.fft.ifft2(fft)
        salencyMap = f.abs()**2
        padded_Mp = torch.nn.ReflectionPad2d(2)(salencyMap)
        salencyMap = torch.nn.functional.conv2d(padded_Mp, gk)
        salMap = salencyMap[0].squeeze().detach().cpu().numpy()
        print(salMap.max(), salMap.min())
        # Display the Result.
        img = b[0].permute([1, 2, 0]).detach().cpu().numpy()
        plt.subplot(1, 2, 1), plt.imshow(img, 'gray')
        #Min max
        salMap -= salMap.min()
        salMap /= salMap.max()
        plt.subplot(1, 2, 2), plt.imshow(salMap, 'gray')
        plt.show()
    end = time.time()
    print(f"time : {end-start} / per image : {(end-start)/len(dataset)}")
if __name__ == '__main__':
    main()