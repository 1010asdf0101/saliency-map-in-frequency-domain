import torch, random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import cv2, time
from pathlib import Path
from torch.utils.data import DataLoader
import tools

device = torch.device('cuda:3')
tools.device = device
import sys
name = sys.argv[1]
target_p = '/data/projects/'+name
paths = list(Path(target_p).rglob('*.jpg'))
paths.extend(list(Path(target_p).rglob("*.JPEG")))
paths.extend(list(Path(target_p).rglob("*.jpeg")))
paths.extend(list(Path(target_p).rglob("*.png")))
paths.extend(list(Path(target_p).rglob("*.bmp")))
paths = [p for p in paths if 'RAW' not in str(p)] 
BATCH_SIZE = 128
random.seed(17)
random.shuffle(paths)
paths = paths[:6528]

def main():
    print('total images : ', len(paths))
    dataset = tools.Train_DT(paths)
    train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, collate_fn=tools.collate_fn)
    gk = tools.gaussian_kernel(5, 8)
    gk = gk.reshape((1, 1, 5, 5)).to(device)
    start = time.time()
    for i, b in enumerate(train_loader):
        red = b[:, 0, :, :]
        green = b[:, 1, :, :]
        blue = b[:, 2, :, :]
        R = red - (green+blue)/2
        G = green - (red+blue)/2
        B = blue - (red+green)/2
        Y = (red+green)/2 - torch.abs(red-green)/2 - blue
        RG = R - G
        BY = B - Y
        I = (red+green+blue)/3
        #add motion value
        M = I.mean(dim = 0).repeat((b.shape[0], 1, 1)) # PQFT랑 결과가 같음. ㅠ
        f1 = M+1j*I
        f2 = BY+1j*RG
        F1 = torch.fft.fft2(f1)
        F2 = torch.fft.fft2(f2)
        mag = torch.sqrt(F1.abs()**2 + F2.abs()**2)
        #remain phase
        F1/=mag
        F2/=mag
        f1 = torch.fft.ifft2(F1)
        f2 = torch.fft.ifft2(F2)
        #saliency map proposed by paper S = g*|q'(t)|**2
        mag = f1.abs()**2+f2.abs()**2
        mag = mag.unsqueeze(1)
        mag = torch.nn.ReflectionPad2d(2)(mag)
        salencyMap = torch.nn.functional.conv2d(mag, gk)
        salMap = salencyMap[0].squeeze().detach().cpu().numpy()
        # Display the Result.
        img = b[0].permute([1, 2, 0]).detach().cpu().numpy()
        plt.subplot(1, 2, 1), plt.imshow(img)
        plt.subplot(1, 2, 2), plt.imshow(salMap)
        plt.savefig(f"results_{name.split('/')[-1]}/batch{i}_PQm.png", pad_inches=0, bbox_inches='tight')
    end = time.time()
    print(f"time : {end-start} / per image : {(end-start)/len(dataset)}")
if __name__ == '__main__':
    main()