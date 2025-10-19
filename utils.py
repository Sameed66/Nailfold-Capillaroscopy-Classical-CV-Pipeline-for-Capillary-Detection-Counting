import cv2
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import os
from skimage import measure
import pandas as pd
from scipy.ndimage import binary_opening
from skimage.morphology import square
from skimage import filters
from skimage import exposure

# Rotation 

def rotate_to_horizontal(image, line_angle):
    rotation_angle = 90+line_angle
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated_image, rotation_matrix 


# Calculating ROI 

def get_roi(image, line, image_path):
    x1, y1, x2, y2 = line[0]
    start = min(x1, x2)
    end = max(x1, x2)
    y_line = min(y1, y2)
    
    # Check if the image name starts with 'S2' or 'S3'
    if os.path.basename(image_path).startswith(('S2', 'S3')):
        height = 150
    else:
        height = 210

    if y_line < height:
        print("La línea está demasiado cerca del borde superior de la imagen para este tamaño de ROI.")
        height -= 10  # Restamos 10 al height si la línea está muy cerca del borde
        if y_line < height:  # Comprobamos nuevamente para asegurarnos de que aún tenemos suficiente espacio
            print("Incluso después de ajustar, la línea sigue estando demasiado cerca del borde.")
            return None
    roi = image[y_line-height:y_line, start-20:end+20]
    return roi

# Preprocessig 

def calculate_luminance(image):
    return image.mean()