import argparse
from PIL import Image
import os
import sys
sys.path.append('../pymeanshift')
import pymeanshift as pms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
import random
import datetime

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

def segment(img, level):
    spatial_radius=6
    range_radius=4.5
    min_density=50

    (segmented_image, labels_image, number_regions) = pms.segment(img,
                                                                spatial_radius,
                                                                range_radius,
                                                                min_density
                                                    )
    # print(segmented_image.shape)
    # print(labels_image)
    # print(number_regions)

    props = measure.regionprops(labels_image, intensity_image=img)

    # filter regions based on size and shape
    min_area = 100  # minimum area of an object in pixels
    max_area = 100000
    max_eccentricity = 0.8  # maximum eccentricity of an object
    if (level == 1):
        min_area = 50  # minimum area of an object in pixels
        max_area = 100000
        max_eccentricity = 0.8  # maximum eccentricity of an object
    if (level == 2):
        min_area = 25  # minimum area of an object in pixels
        max_area = 100
        max_eccentricity = 0.8  # maximum eccentricity of an object
        
    objects = []
    for i, prop in enumerate(props):
        if prop.area >= min_area and prop.area <= max_area and prop.eccentricity <= max_eccentricity:
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
    # for obj in objects:
    #     print(f"Object {obj['label']}: area = {obj['area']}, centroid = {obj['centroid']}, mean intensity = {obj['mean_intensity']}, bbox = {obj['bbox']}")
    return labels_image, objects

def random_array(nums_of_spots, n):
    random_spots = []

    for i in range (nums_of_spots):
        random_spots.append(random.randint(0, n))
        # 0 for remove, 1 for change color, 2 for rotate

    return random_spots

def take_random_objs(objects, nums_of_spots):
    return random.sample(objects, nums_of_spots)

def get_background_color(img):
    return img[0][0]

def get_random_color():
    # Generate a random RGB color tuple
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    color = (r, g, b)

    # Print the color tuple
    return color


def main():
    args = parse_args()

    #read input image and segment it into objects
    input_img = cv2.imread(args.input_image)
    labels_image, objects = segment(input_img, args.level)
    #define number of differences modified for image
    nums_of_spots = int(args.nums_of_spots)

    #create random spots and pick random modify for each spot
    random_modify = random_array(nums_of_spots, 2) #3 modify actions
    random_objs = take_random_objs(objects, nums_of_spots)

    # print(random_modify)
    # for obj in random_objs:
    #     print(obj['label'], ' ')
    
    background_color = get_background_color(input_img)

    if not os.path.exists('output'):
        os.makedirs('output')
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    output_dir = os.path.join(args.output, now)
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_img = input_img.copy()
    for i, obj in enumerate (random_objs):
        label_to_change = obj['label']
        modify = random_modify[i]
        object_color = background_color if modify == 0 else get_random_color()

        # Set the pixel values of the object to the desired color using NumPy indexing
        output_img[labels_image == (label_to_change+1)] = object_color
        img_id = datetime.datetime.now().strftime('%f')

    cv2.imwrite(output_dir+'/'+img_id+'.png', output_img)

if __name__ == '__main__':
    main()
