import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def info(org):
    # highboost
    k = 2
    gauss = cv2.GaussianBlur(org, (7,7), 0)
    hbf = cv2.addWeighted(org, k+1, gauss, -k, 0)
    
    # adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3,tileGridSize=(5,5))
    histEq = clahe.apply(hbf)
    
    # median filtering
    median = cv2.medianBlur(histEq , 3)
    
    # morph gradient
    kernel = np.ones((3,3),np.uint8)
    gradient = cv2.morphologyEx(median, cv2.MORPH_GRADIENT, kernel)

    # canny
    edges = cv2.Canny(gradient,110,255)
    
    # otsu
    blur = cv2.GaussianBlur(edges,(3,3),0)
    _,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    plt.clf()
    plt.subplot(231) , plt.title("Original Image") , plt.imshow(org , cmap="gray")
    plt.subplot(232) , plt.title("Highboost Filtering (k=2)") , plt.imshow(hbf , cmap="gray")
    plt.subplot(233) , plt.title("Adaptive Histogram Equalization") , plt.imshow(histEq , cmap="gray")
    plt.subplot(234) , plt.title("Morphological Gradient") , plt.imshow(gradient , cmap="gray")
    plt.subplot(235) , plt.title("Canny Edge Detection") , plt.imshow(edges , cmap="gray")
    plt.subplot(236) , plt.title("Otsu's Thresholding") , plt.imshow(thresh , cmap="gray")
#     plt.savefig("/home/ojas3/Desktop/xx.png")
    plt.show()

if __name__ == '__main__':
    img = cv2.imread("../train/NORMAL/NORMAL2-IM-0890-0001.jpeg" , 0)
    info(img)