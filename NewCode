import matplotlib.pyplot as plt
import numpy as np
import skimage.io as im
import pathlib
import os
import cv2
from statistics import mean
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import csv

directories = ["/home/jovyan/data/img/codex_sample/src_data/Cyc1_reg1", "/home/jovyan/data/img/codex_sample/src_data/Cyc2_reg1"]

class sections:
    
    def __init__(self, start, end):
        self.value = start
        self.end = end
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.value > self.end:
            raise StopIteration
        i = [0,0,0,1,1,1,2,2,2][self.value - 1]
        j = [0,1,2][(self.value - 1) % 3]
        self.value += 1
        return i, j

def findImages(directories, lst):
    global opath
    f = 0
    i = 0
    for r in range(len(directories)):
        directory = directories[r]
        for path in pathlib.Path(directory).iterdir(): #Finding images in directory
            if path.is_file():
                opath = open(path, 'r').name
                if ".tif" in opath:
                    image = im.imread(opath, plugin="tifffile")
                    if sum(image[0])/len(image) > 1: #Mean of first line of photo, blank images are ignored
                        lst.append(image)
                        i += 1
                print(f"LOADING IMAGES ({i} out of {f} files from {len(directories)} directories)", end="\r")
            f += 1
    return lst

def signalNoise(img):
    axis = 0
    ddof = 0
    Arr = np.asanyarray(img)
    me = Arr.mean(axis)
    sd = Arr.std(axis=axis, ddof=ddof)
    SNR = np.where(sd == 0, 0, me/sd)
    return SNR

def imageBlur(img):
    learning_data = pd.read_csv("TrainingSet.csv")
    X = learning_data.drop(columns="BLUR") #Input data set
    y = learning_data["BLUR"] #Output data set
    model = DecisionTreeClassifier()
    model.fit(X, y)
    
    third = int(round(len(img)/3))
    blur = []
    
    for s in sections(1, 9):
        sect = [[None for i in range(third)] for j in range(third)]
        for y in range(third):
            for x in range(third):
                sect[y][x] = img[y+(s[0]*third)][x+(s[1]*third)]
        blur.append(round(cv2.Laplacian(np.array(sect), cv2.CV_64F).var()/1000)) #Laplacian on each section of image

    blur = float(model.predict([[min(blur), max(blur)]])/10) #Machine learning
    return blur

print("LOADING IMAGES", end="\r")
loi = findImages(directories, [])
print("\n")

for i in range(len(loi)): #Main loop len(loi)
    print(f"\033[1;34;49m{opath.split('/')[-1]}\033[1;30;49m")
    image = loi[i]
    
    plt.imshow(image) #Display image
    plt.show()
    
    SNR = signalNoise(image)
    print("Signal noise:")
    plt.plot(SNR) #Plot signal-to-noise ratio
    plt.show()
    
    blur_value = imageBlur(image) #Find blurriness if image is blurry
    print("Image blur:", blur_value)
    
    print()
print("\nDONE")
f.close()
