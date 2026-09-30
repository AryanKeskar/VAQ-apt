import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


path = "/Users/home_folder/Desktop/Python_projects/VAQ-apt/scripts/temp/entropy_method.png"
#the image was saved as a BGR image, so we need to read it as RGB
image = cv2.imread(path, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_name = "rgb_entropy_method.png"
# Save the image as a PNG file at the same path as the original image
cv2.imwrite(image_name, image)
