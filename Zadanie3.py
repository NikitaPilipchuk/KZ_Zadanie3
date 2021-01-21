import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
from skimage.filters import try_all_threshold
from skimage.filters import threshold_otsu, threshold_triangle 
from skimage.measure import label, regionprops


def to_gray(image):
    return (image[:,:,0] * 0.2126 + image[:,:,1] * 0.7152 + image[:,:,2] * 0.0722).astype("uint8")

def binarisation(image, limit):
    B = image.copy()
    B[B > limit] = 0
    B[B > 0] = 1
    return B

def hist(gray):
    H = np.zeros(256)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            val = gray[i,j]
            H[val] += 1
    return H

def recognize_objects(image):
    rgb_bin = np.copy(image)
    
    for i in [0,1,2]:
        thresh = threshold_otsu(image[:,:,i])
        rgb_bin[:,:,i] = binarisation(image[:,:,i], thresh + 20) 
        
    binary = rgb_bin[:,:,0] + rgb_bin[:,:,1] + rgb_bin[:,:,2]
    binary[binary > 0] = 1
    binary = morphology.binary_opening(binary, iterations = 2)
     
    labeled = label(binary)
    
    areas = []
    
    regions = regionprops(labeled)
    areas = [region.area for region in regions]
    
    limit = 0.3 * np.std(areas)
    
    for i, region in enumerate(regionprops(labeled)):
        if region.bbox[0] == 0 or region.bbox[1] == 0  or region.bbox[2] == labeled.shape[0]-1  or region.bbox[3] == labeled.shape[1]-1:
            circ = region.perimeter ** 2 / region.area
            if circ > 60:
                labeled[labeled == region.label] = 0
        if region.area < limit:
            labeled[labeled == region.label] = 0
            
    binary = labeled.copy()
    binary[binary > 0] = 1
    
    return label(binary)    

def count_pencils(labeled):
    regions = regionprops(labeled)
    return sum(i.eccentricity > 0.98 for i in regions)

def print_answer(directory, details = False):
    tree = os.walk(directory)
    num_obj = 0
    num_pen = 0
    for address, dirs, files in tree:
        for file in files:
            image = plt.imread(f"{address}/{file}")
            labeled = recognize_objects(image)
            pencils = count_pencils(labeled)
            num_obj += np.max(labeled)
            num_pen += pencils
            if details:
                print(f'На изображении "{file}" объектов: {np.max(labeled)}, из них карандашей: {pencils}')
                
    print(f'\nРезультат для каталога "{directory}":\n')
    if details: print(f"Суммарное количество объектов: {num_obj}")
    print(f"Суммарное количество карандашей: {num_pen}")
    
print_answer("images", True)
