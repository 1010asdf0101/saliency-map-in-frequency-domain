import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
WIDTH = 256   # has a great influence on the result

img = cv2.imread('flower.png', 0)
img = cv2.resize(img, (WIDTH,WIDTH*img.shape[0]//img.shape[1]))

#c = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
c = np.fft.fft2(np.float32(img))
print(c[0, 0])
#mag = np.sqrt(c[:,:,0]**2 + c[:,:,1]**2)
mag = np.abs(c)
kernel = np.ones((3, 3), dtype = np.float32) / 9.
spectralResidual = np.exp(np.log(mag) - cv2.filter2D(np.log(mag), -1, kernel))
c.real = c.real * spectralResidual / mag
c.imag = c.imag * spectralResidual / mag
#c[:,:,0] = c[:,:,0] * spectralResidual / mag
#c[:,:,1] = c[:,:,1] * spectralResidual / mag
#c = cv2.dft(c, flags = (cv2.DFT_INVERSE | cv2.DFT_SCALE))
c = np.fft.ifft2(c)
#mag = c[:,:,0]**2 + c[:,:,1]**2
mag = np.abs(c)**2
#cv2.normalize(cv2.GaussianBlur(mag,(9,9),3,3), mag, 0., 1., cv2.NORM_MINMAX)
plt.imshow(mag, 'gray')
plt.show()