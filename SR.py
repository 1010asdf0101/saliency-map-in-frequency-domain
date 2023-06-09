import torch, random
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
    dataset = tools.Train_DT(paths, is_gray=True)
    train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, collate_fn=tools.collate_fn)
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32) / 9.
    kernel = kernel.to(device)
    gk = tools.gaussian_kernel(5, 8)
    gk = gk.reshape((1, 1, 5, 5)).to(device)
    start = time.time()
    for i, b in enumerate(train_loader):
        fft = torch.fft.fft2(b)
        mag = torch.abs(fft) # amplitude -> A
        La = torch.log(mag) # log amplitude
        padded_La = torch.nn.ReflectionPad2d(1)(La)
        avgLogAmp = torch.nn.functional.conv2d(padded_La, kernel)
        SR = torch.exp(La - avgLogAmp)
        #Q = A*e^(i*P) * SR / mag = SR * e^(i*P)
        fft.real = fft.real * SR / mag
        fft.imag = fft.imag * SR / mag
        f = torch.fft.ifft2(fft)
        salencyMap = f.abs()**2
        padded_Mp = torch.nn.ReflectionPad2d(2)(salencyMap)
        salencyMap = torch.nn.functional.conv2d(padded_Mp, gk)
        salMap = salencyMap[0].squeeze().detach().cpu().numpy()
        #min-max scaling
        salMap -= salMap.min()
        salMap /= salMap.max()
        # Display the Result.
        img = b[0].permute([1, 2, 0]).detach().cpu().numpy()
        plt.subplot(1, 2, 1), plt.imshow(img, 'gray')
        plt.subplot(1, 2, 2), plt.imshow(salMap, 'gray')
        plt.savefig(f"results_{name.split('/')[-1]}/batch{i}_SR.png", pad_inches=0, bbox_inches='tight')
    end = time.time()
    print(f"time : {end-start} / per image : {(end-start)/len(dataset)}")
if __name__ == '__main__':
    main()