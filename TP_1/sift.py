import cv2 as cv
import numpy as np



def sift(image):

    sift = cv.SIFT_create()
    keypoints, descs  = sift.detectAndCompute(image, None)
    image_with_keypoints =  cv.drawKeypoints(image, keypoints, outImage=None)
    # cv.imshow("Images ", image_with_keypoints)
    
    return image_with_keypoints


# Octace with gaussian filter- kernel - dog

def octave_gaussian(image, n=4, level=3):
    
    num_octave = n
    scale_factore = 2
    gauss_step = np.linspace(1, 2, level)
    pyramid = {}
    for x in range(num_octave):
        pyramid["level"+str(x)] = [image]
        i = 0
        temp = []
        for k in range(level):
            temp.append(gaussian_kernel(image, gauss_step[i]))
            i += 1
        pyramid["level"+str(x)] = temp
        image = cv.resize(image, None, fx=1/scale_factore, fy=1/scale_factore)
    return pyramid

def gaussian_kernel(image, n):
    return cv.GaussianBlur(image, None, n)

def DOG_pyramid(pyramid, num_octave):
    dog_pyramid = {}
    for k in range(num_octave):
        temp = []
        for i in range(len(pyramid["level"+str(k)]) -  1):
            temp.append(pyramid["level"+str(k)][i] - pyramid["level"+str(k)][i + 1])
        dog_pyramid["level"+str(k)] = temp

    return dog_pyramid


def find_extrema(dog_image, threshold):
    # Utilisez la fonction cv2.findLocalMaxima pour trouver les maxima locaux
    local_maxima = cv.dnn_superres.findLocalMaxima(dog_image, threshold)

    return local_maxima





    
