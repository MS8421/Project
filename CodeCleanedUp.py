from matplotlib.pyplot import imshow as plotArrayAsImage
from matplotlib.pyplot import show as showImage
import numpy
import skimage.io
import skimage.measure
import pathlib
import os


1)build a represeantation of dataset
  - find directories
     - go through the files and find images

2) read images
3) calculate metrics
4) write results to the disk
        
def main():
    listing = get_dir_structure(dataset_path)
    #read_images_from_listing(listing)
    #process_images(listing)
    for directory in directories:
        for image_path in directories[directory]:
            img = read_img(image_path)   
            metric_resutls = calc_metrics(img)


    
    
  

for directory in directories:
    for image_path in directories[directory]:
        read(image_path)

      
  

def findImagesFromDirectories(directories: List[Path]) -> List[Path]:
    img_path_list = []
    for i in range(len(directories)):
        currentDirectory = directories[i]
        for path in pathlib.Path(currentDirectory).iterdir(): #Finding images in current directory
            is_image = check_if_img_file(path)
            if is_image:
                img_path_list.append(path)
    return img_path_list

def load_imgs(img_path_list: List[Path]) -> List[np.ndarray]:
    img_list = []
    for img_path in img_path_list:
        img = skimage.io.imread(img_path, plugin="tifffile")
        img_list.append(img)
    return img_list


def filterFiles(path, listOfImages):
    if path.is_file():
        openPath = open(path, 'r').name
        if ".tif" in openPath:
            currentImage = skimage.io.imread(openPath, plugin="tifffile")
            if sum(currentImage[0])/len(currentImage) > 1: #Mean of first line of photo, blank images are ignored
                loadImageToArray(currentImage, listOfImages)

def loadImageToArray(currentImage, listOfImages):
    listOfImages.append(currentImage)
    print(f"Loading {len(listOfImages)} images from {len(directories)} directories...", end="\r")

def calculateSignalNoise(img):
    meanOfImage = numpy.mean(img)
    standardDeviationOfImage = numpy.std(img)
    signalNoiseRatio = (meanOfImage**2 / standardDeviationOfImage**2 )
    return signalNoiseRatio

def calculateImageBlur(img):
    return skimage.measure.blur_effect(img)

directories = ["/home/jovyan/data/img/codex_sample/src_data/Cyc1_reg1"] #, "/home/jovyan/data/img/codex_sample/src_data/Cyc2_reg1"]
listOfImages = findImagesFromDirectories(directories, [])
print("\n")
for i in range(len(listOfImages)): #Main loop
    currentImage = listOfImages[i]
    
    plotArrayAsImage(currentImage) #Display the current image
    showImage()

    print("Signal noise:", calculateSignalNoise(currentImage)) #print the signal noise ratio
    print("Image blur:", calculateImageBlur(currentImage)) #print the blur effect
    
    print()
print("\nDONE")
