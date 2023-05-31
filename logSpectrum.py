import torch
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

def main():
    BATCH_SIZE = 128
    dataset = Train_DT(paths)
    train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    resizer = torchvision.transforms.Resize((256, 256), antialias=True)
    start = time.time()
    for b in train_loader:
        b = resizer(b)
        h, w = b.shape[-2:]
        fft = torch.fft.fft2(b)
        logAmplitude = torch.log(fft.abs())
        batch_mean = torch.mean(logAmplitude.squeeze(), dim = 0)
        area = batch_mean[:h//2, :w//2]
        aa = torch.sum(area, dim=0)
        aa = aa.detach().cpu().numpy()
        plt.title(f"{BATCH_SIZE}")
        plt.plot(aa)
        plt.show()
    end = time.time()
    print(f"time : {end-start} / per image : {(end-start)/len(dataset)}")
if __name__ == '__main__':
    main()