import os
import sys
# from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io
from skimage.transform import resize
import numpy as np


def crop_and_resize_images(path, new_path, cropx, cropy, img_size=256):
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    dirs = [l for l in os.listdir(path) if l != '.DS_Store']
    total = 0

    for item in dirs:
        img = io.imread(path + item)
        x, y, channel = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        img = img[startx:startx + cropx, starty:starty + cropy]
        img = resize(img, (img_size, img_size))
        io.imsave(str(new_path + item), img)
        total += 1
        print("Saving: ", item, total)


if __name__ == '__main__':
    crop_and_resize_images(path='../data/train/', new_path='../data/train-resized-256/', cropx=1800, cropy=1800,
                           img_size=256)
    crop_and_resize_images(path='../data/test/', new_path='../data/test-resized-256/', cropx=1800, cropy=1800,
                           img_size=256)
