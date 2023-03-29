from PIL import Image
import cv2
from matplotlib import pyplot as plt
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils

def threshold(tup1, tup2):
    val=0
    if(abs(tup1[0]-tup2[0])>=25):
        val+=1
    if(abs(tup1[1]-tup2[1])>=25):
        val+=1
    if(abs(tup1[2]-tup2[2])>=25):
        val+=1

    if(val>=2):
        return True
    else:
        return False


def main(IMG1, IMG2):

    # resultImg.save('result.png')
    detect_differences(IMG1, IMG2)


def detect_differences(img1_path, img2_path, threshold=10, min_blob_size=0):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Convert images to grayscale and blur them
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    blurred1 = cv2.GaussianBlur(gray1, (11, 11), 0)
    blurred2 = cv2.GaussianBlur(gray2, (11, 11), 0)

    # Compute absolute difference between the images and threshold the result
    diff = cv2.absdiff(blurred1, blurred2)

    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to clean up the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=4)

    # Label and filter blobs in the thresholded image
    labels = measure.label(thresh, connectivity=2, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > min_blob_size:
            mask = cv2.add(mask, labelMask)

    # Find contours of the blobs and draw circles around them on the original images
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(img1, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)
        cv2.circle(img2, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)
    res = np.concatenate((img1, img2), axis=1)
    cv2.imwrite('result.png', res)
    # Display the resulting images
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(300, 100))
    axes[0].imshow(img1[...,::-1])
    axes[1].imshow(img2[...,::-1])
    plt.show()

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description='The script will spot the differences between two images, circle and number them. ')
    my_parser.add_argument('--img1', action='store', type=str, required=True, help='the path to image 1')
    my_parser.add_argument('--img2', action='store', type=str, required=True, help='the path to image 2')
    # IMG1 = input('Enter path to image 1: ')
    # IMG2 = input('Enter path to image 2: ')
    args = my_parser.parse_args()
    IMG1 = args.img1 
    IMG2 = args.img2
    main(IMG1, IMG2)