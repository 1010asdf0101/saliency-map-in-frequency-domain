import torch
import numpy as np
import torch, torchvision
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import cv2, time
from pathlib import Path
from torch.utils.data import DataLoader
import tools

device = torch.device('cuda:3')
tools.device = device
target_p = '/data/projects/'
paths = list(Path(target_p).rglob('*.jpg'))
paths.extend(list(Path(target_p).rglob("*.JPEG")))
paths.extend(list(Path(target_p).rglob("*.jpeg")))
paths.extend(list(Path(target_p).rglob("*.png")))
paths.extend(list(Path(target_p).rglob("*.bmp")))
BATCH_SIZE=128

def main():
    dataset = tools.Train_DT(paths)
    train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, collate_fn=tools.collate_fn)
    kss = [i for i in range(1, int(np.ceil(np.log2(128))+1), 2)]
    print('kernel sizes : ', kss)
    gk = [torch.nn.ZeroPad2d((kss[-1]-k)//2)(tools.gaussian_kernel(k)) for k in kss]
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
    end = time.time()
    print(f"time : {end-start} / per image : {(end-start)/len(dataset)}")
if __name__ == '__main__':
    main()