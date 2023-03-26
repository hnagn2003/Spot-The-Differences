import argparse
from PIL import Image
import sys
sys.path.append('../pymeanshift')
import pymeanshift as pms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--level',
        default=1,
        help='level of generate')
    parser.add_argument(
        '--input_image',
        default='input/input_image.png',
        help='input image file/url')
    parser.add_argument(
        '--output',
        default='output',
        help='output image file/url')
    parser.add_argument(
        '--nums_of_output',
        default=3,
        help='number of output'
    )
    parser.add_argument(
        '--nums_of_spots',
        default=3,
        help='number of spots'
    )
    args = parser.parse_args()
    return args

def segment(img, args):
    (segmented_image, labels_image, number_regions) = pms.segment(img,
                                                                spatial_radius=6,
                                                                range_radius=4.5,
                                                                min_density=50
                                                    )
    print(segmented_image.shape)
    print(labels_image)
    print(number_regions)

    props = measure.regionprops(labels_image, intensity_image=img)

    # filter regions based on size and shape
    min_area = 50  # minimum area of an object in pixels
    max_eccentricity = 0.8  # maximum eccentricity of an object
    objects = []
    for i, prop in enumerate(props):
        if prop.area >= min_area and prop.eccentricity <= max_eccentricity:
            objects.append({
                'label': i,
                'area': prop.area,
                'centroid': prop.centroid,
                'bbox': prop.bbox,
                'mean_intensity': prop.mean_intensity,
                'min_intensity': prop.min_intensity,
                'max_intensity': prop.max_intensity
            })

    # display or use the object information
    for obj in objects:
        print(f"Object {obj['label']}: area = {obj['area']}, centroid = {obj['centroid']}, mean intensity = {obj['mean_intensity']}, bbox = {obj['bbox']}")
    # cv2.imshow('output', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def random_array(nums_of_spots, nums_of_objects, aim = 'spot'):
    random_spots = []

    for i in range(nums_of_spots):
        if (aim == 'spot'):
            random_spots.append(random.randint(0, nums_of_objects))
        else:
            random_spots.append(random.randint(0, 2)) # 0 for remove, 1 for change color, 2 for rotate

    return random_spots

def main():
    args = parse_args()
    img = cv2.imread(args.input_image)
    objects = segment(img, args)
    nums_of_spots = args.nums_of_spots

    #create random spots and pick random modify for each spot
    random_objs = random_array(nums_of_spots, len(objects))
    random_modify = random_array(nums_of_spots, len(objects), 'modify')

    for i, obj in enumerate (objects):
        if (random_modify[i] == 0): #remove
            #do sth
    

if __name__ == '__main__':
    main()
