import numpy as np
import torch, torchvision, random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import cv2, time
from pathlib import Path
from torch.utils.data import DataLoader
import tools

dt = torch.float64 #64 bit float for precise calculation
device = torch.device('cuda:3')
tools.device = device
target_p = '.'
paths = list(Path(target_p).rglob('*.bmp'))
#paths.extend(list(Path(target_p).rglob("*.JPEG")))
#paths.extend(list(Path(target_p).rglob("*.jpeg")))
#paths.extend(list(Path(target_p).rglob("*.png")))
#paths.extend(list(Path(target_p).rglob("*.bmp")))
paths = [p for p in paths if 'RAW' not in str(p)] 
BATCH_SIZE = 128
random.seed(17)
random.shuffle(paths)
paths = paths[:6528]

def gaussian_kernel(size, t0=0.5):
    kernel = torch.Tensor(size, size)
    center = size // 2
    sigma = (2**(size-1))*t0
    variance = sigma ** 2
    coefficient = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    for i in range(size):
        for j in range(size):
            distance = (i - center)**2 + (j - center)**2
            kernel[i, j] = coefficient * np.exp(-distance / (2 * variance))
    kernel = kernel / torch.sum(kernel)
    return kernel


def main():
    BATCH_SIZE = max(1, len(paths)//50)
    softmax = torch.nn.Softmax2d()
    ref_padder = torch.nn.ReflectionPad2d(2)
    dataset = tools.Train_DT(paths)
    train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, collate_fn=tools.collate_fn)
    # build gaussian kernels with various size
    kss = [i for i in range(1, int(np.ceil(np.log2(128))+1), 2)]
    print('kernel sizes : ', kss)
    gk = [torch.nn.ZeroPad2d((kss[-1]-k)//2)(gaussian_kernel(k).type(dt)) for k in kss]
    gk = torch.stack(gk).unsqueeze(1).to(device)
    kernel = gaussian_kernel(5).type(dt)
    kernel = kernel.repeat((len(kss), 1, 1, 1)).to(device)
    start = time.time()
    for b in train_loader:
        # scaling to [0, 1]
        img = b[0]
        b = b.type(dt)
        red = b[:, 0, :, :]/255
        green = b[:, 1, :, :]/255
        blue = b[:, 2, :, :]/255
        # calulating Lab
        R = red - (green+blue)/2
        G = green - (red+blue)/2
        B = blue - (red+green)/2
        Y = (red+green)/2 - torch.abs(red-green)/2 - blue
        RG = R - G
        BY = B - Y
        I = (red+green+blue)/3
        # make quaternion
        f1 = 1j*I
        f2 = BY+1j*RG
        F1 = torch.fft.fft2(f1)
        F2 = torch.fft.fft2(f2)
        mag = torch.sqrt(F1.abs()**2 + F2.abs()**2)
        F1/=mag
        F2/=mag
        mag.unsqueeze_(1)
        # apply gaussian kernel
        mag = torch.nn.ReflectionPad2d(gk.shape[-1]//2)(mag)
        lam = torch.nn.functional.conv2d(mag, gk, groups=1) # F.conv2d kernel shape=(out_ch, in_ch/grp, kH, kW)
        Lambda1 = lam*F1.repeat(1, len(kss), 1, 1)
        Lambda2 = lam*F2.repeat(1, len(kss), 1, 1)
        f1 = torch.fft.ifft2(Lambda1)
        f2 = torch.fft.ifft2(Lambda2)
        mag = torch.sqrt(f1.abs()**2 + f2.abs()**2)
        mag = ref_padder(mag)
        salency_set = torch.nn.functional.conv2d(mag, kernel, groups=len(kss))
        # calculating entropy 2D
        p_sal = softmax(salency_set)
        p_sal = ref_padder(p_sal) # (B, C, H, W)
        gk = tools.gaussian_kernel(5, 0.03).type(dt)
        gk = gk.repeat((len(kss), 1, 1, 1)).to(device)
        p_sal = torch.nn.functional.conv2d(p_sal, gk, groups=len(kss))
        entropy = torch.sum(-p_sal*torch.log(p_sal), dim=(2, 3))
        # calculating lambda_k
        summation = torch.sum(salency_set, dim=(2, 3)).reshape((1, len(kss), 1, 1))
        normalizedSal = salency_set/summation
        gk = tools.gaussian_kernel(tools.h, tools.h/4).to(device)
        gk = gk.repeat(1, 4, 1, 1)
        lambda_k = gk * normalizedSal
        lambda_k = 1/lambda_k.sum(dim = (2, 3))
        res = entropy*lambda_k
        properK = torch.argmin(res, dim = 1)
        #show images
        img = img.permute([1, 2, 0]).detach().cpu().numpy().astype(np.uint8)
        sal = salency_set[0, properK]
        sal -= sal.min()
        sal /= sal.max()
        sal *= 255
        sal = sal.permute([1, 2, 0]).detach().cpu().numpy().astype(np.uint8)
        plt.title(f"Kernel Size : {properK}")
        plt.subplot(1, 2, 1), plt.imshow(img)
        plt.subplot(1, 2, 2), plt.imshow(sal)
        plt.show()        
    end = time.time()
    print(f"time : {end-start} / per image : {(end-start)/len(dataset)}")
if __name__ == '__main__':
    main()