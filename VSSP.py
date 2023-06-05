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
#paths = Path('/home/shawnman99/data/cream/OSP/OSP_decay').rglob('*.png')
paths = Path('/home/shawnman99/img_intern/sin3/').rglob('*.bmp')
BATCH_SIZE=16
## build dataloader  &  define required functions
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

class Train_DT(Dataset):
    def __init__(self, paths):
        trans = torchvision.transforms.PILToTensor()
        resizer = torchvision.transforms.Resize((128, 128), antialias=True)
        self.items=[]
        for p in paths:
            img = Image.open(str(p)).convert('RGB')
            img = resizer(trans(img))
            self.items.append(img)
    def __len__(self):
        return len(self.items)
    def __getitem__(self, index):
        return self.items[index]
    
def collate_fn(batch):
    ret = torch.stack(batch).to(device)
    return ret

def gaussian_kernel(size, t0=0.5):
    kernel = torch.Tensor(size, size)
    sigma = math.ceil(2**(size-1)*t0)
    center = size // 2
    variance = sigma**2
    coefficient = 1.0 / (math.sqrt(2 * math.pi)*sigma)
    for i in range(size):
        for j in range(size):
            distance = (i - center)**2 + (j - center)**2
            kernel[i, j] = coefficient * math.exp(-distance / (2 * variance))
    kernel = kernel / torch.sum(kernel)
    return kernel

def main():
    dataset = Train_DT(paths)
    train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    kss = [i for i in range(1, int(math.ceil(math.log2(128))+1), 2)]
    print('kernel sizes : ', kss)
    gk = [torch.nn.ZeroPad2d((kss[-1]-k)//2)(gaussian_kernel(k)) for k in kss]
    gk = torch.stack(gk).unsqueeze(1).to(device)
    start = time.time()
    for b in train_loader:
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
        alpha, beta, gamma = 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)
        F1 = torch.fft.fft2(I)
        F2 = torch.fft.fft2(RG)
        F3 = torch.fft.fft2(BY)
        aa = -alpha*F1.imag - beta*F2.imag - gamma*F3.imag
        bb = F1.real + gamma*F2.imag - beta*F3.imag
        cc = F2.real + alpha*F3.imag - gamma*F1.imag
        dd = F3.real + beta*F1.imag - alpha*F2.imag
        mag = torch.sqrt(aa**2 + bb**2 + cc**2 + dd**2)
        #normalize
        aa/=mag
        bb/=mag
        cc/=mag
        dd/=mag
        mag.unsqueeze_(1)
        mag = torch.nn.ReflectionPad2d(gk.shape[-1]//2)(mag)
        lam = torch.nn.functional.conv2d(mag, gk, groups=1) # F.conv2d(out, in/gr, kH, kW)
        print(lam.shape)
        for test in lam[0]:
            ta = aa*test
            tb = bb*test
            tc = cc*test
            td = dd*test
            fa = torch.fft.ifft2(ta)
            fb = torch.fft.ifft2(tb)
            fc = torch.fft.ifft2(tc)
            fd = torch.fft.ifft2(td)
            f1 = fa.real - alpha*fb.imag - beta*fc.imag - gamma*fd.imag
            f2 = fb.real + alpha*fa.imag + gamma*fc.imag - beta*fd.imag
            f3 = fc.real + beta*fa.imag + alpha*fd.imag - gamma*fb.imag
            f4 = fd.real + gamma*fa.imag +beta*fb.imag - alpha*fc.imag
            #saliency map proposed by paper S = g*|q'(t)|**2
            mag = f1**2+f2**2+f3**2+f4**2
            mag = mag[0]
            salMap = mag.detach().cpu().numpy()
            salMap = cv2.GaussianBlur(salMap, (5, 5), 8)
            # Display the Result.
            img = b[0].permute([1, 2, 0]).detach().cpu().numpy()
            print(img.shape, salMap.shape)
            plt.subplot(1, 2, 1), plt.imshow(img)
            plt.subplot(1, 2, 2), plt.imshow(salMap)
            plt.show()
        exit()
        fa = torch.fft.ifft2(aa)
        fb = torch.fft.ifft2(bb)
        fc = torch.fft.ifft2(cc)
        fd = torch.fft.ifft2(dd)
        F1 = fa.real - alpha*fb.imag - beta*fc.imag - gamma*fd.imag
        F2 = fb.real + alpha*fa.imag + gamma*fc.imag - beta*fd.imag
        F3 = fc.real + beta*fa.imag + alpha*fd.imag - gamma*fb.imag
        F4 = fd.real + gamma*fa.imag +beta*fb.imag - alpha*fc.imag
        #saliency map proposed by paper S = g*|q'(t)|**2
        mag = F1**2+F2**2+F3**2+F4**2
        mag = mag.unsqueeze(1)
        mag = torch.nn.ReflectionPad2d(2)(mag)
        salencyMap = torch.nn.functional.conv2d(mag, gk)
        salMap = salencyMap[0].squeeze().detach().cpu().numpy()
        # Display the Result.
        img = b[0].permute([1, 2, 0]).detach().cpu().numpy()
        plt.subplot(1, 2, 1), plt.imshow(img)
        plt.subplot(1, 2, 2), plt.imshow(salMap)
        plt.show()
    end = time.time()
    print(f"time : {end-start} / per image : {(end-start)/len(dataset)}")
if __name__ == '__main__':
    main()