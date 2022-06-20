import matplotlib.pyplot as plt
import numpy as np
import skimage.io as im
import pathlib
import os
import cv2
from typing import List, Tuplt, Dict, Any
from statistics import mean
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import csv
import dask

directories = ["/home/jovyan/data/img/codex_sample/src_data/Cyc1_reg1", "/home/jovyan/data/img/codex_sample/src_data/Cyc2_reg1"]

Image = np.ndarray

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

slice(0, 10, 1)
slice(None)
img[(slice(0, 10), slice(0, 10))]


def findImages(directories, lst) -> List[Image]:
    global opath
    f = 0
    i = 0
    for r in range(len(directories)):
        directory = directories[r]
        for path in pathlib.Path(directory).iterdir(): #Finding images in directory
            if path.is_file():
                opath = path.name
                if opath.suffix in (".tif", ".tiff"):
                    image = im.imread(opath, plugin="tifffile")
                    if sum(image[0])/len(image) > 1: #Mean of first line of photo, blank images are ignored
                        lst.append(image)
                        i += 1
                print(f"LOADING IMAGES ({i} out of {f} files from {len(directories)} directories)", end="\r")
            f += 1
    return lst

{"Cyc1_reg1": {"1_00001_Z001_CH1.tif": {"snr": 1.0, "bluriness": 1232312391}}}}


def read_and_process_img(img_path: Path) -> Dict[str, Dict[str, float]:
    image = im.imread(img_path, plugin="tifffile")
    results = calc_metrics(image)
    fin_res = {img_path.name: results}
    return fin_res

def read_and_process_imgs_parallelized(dataset_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]:
    fin_res = dict()
    for img_dir in img_dirs:
        tasks = []
        for img_path in img_dirs[img_dir]:
            task = dask.delayed(read_and_process_img)(img_path)
            tasks.append(task)
        this_dir_results = dask.compute(*tasks)
        fin_res[img_dir] = this_dir_results
    return fin_res
     
                                                                 
def signalNoise(img: Image) -> float:
    mean: np.ndarray = np.mean(img)
    std = np.std(img)
    snr = (mean**2 / std**2 )
    return snr
    
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

  
                                                                  
Cyc1_reg1
----Channels 1-4
    ------- Z-planes 1-4
     ------------- Tiles 4
Cyc2_reg1
                                                                  
                                                                  
import json

with open("out_file.json", "w") as f:
    json.dump(dictionary, f)

                                                                  
                                                                  

                                                                  
def main(input_path: Path):
    do_proc(input_path)

                                                                  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=Path, help="path to the input directorye with images")
    
    args = parser.parse_args()
    main(args.i)                                                              
                                                                  

                                                                  
                                                                  
                                                                  
#import NewCode
#NewCode.main(path)
                                                                  
                                                                  
                                                                  
                                                                                                                 
                                                                  
                                                                  
