import random

import cv2 as cv
import numpy as np
from imgaug import augmenters as iaa

from config import img_rows, img_cols
from data_generator import get_image, get_category, to_bgr

if __name__ == '__main__':
    with open('names.txt', 'r') as f:
        names = f.read().splitlines()

    filename = 'valid_ids.txt'
    with open(filename, 'r') as f:
        ids = f.read().splitlines()
        ids = list(map(int, ids))
    id = random.choice(ids)
    name = names[id]
    image = get_image(name)
    category = get_category(id)

    image = cv.resize(image, (img_rows, img_cols), cv.INTER_CUBIC)
    category = cv.resize(category, (img_rows, img_cols), cv.INTER_NEAREST)
    category_bgr = to_bgr(category)

    images = np.zeros((1, img_rows, img_cols, 3))
    images[0] = image
    categories = np.zeros((1, img_rows, img_cols, 1))
    categories[0] = category

    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
    ])
    seq_det = seq.to_deterministic()

    images_aug = seq_det.augment_images(images)
    categories_aug = seq_det.augment_images(categories)

    cv.imshow('image', images_aug[0])
    cv.imshow('category_bgr', categories_aug[0])
    cv.waitKey(0)
    cv.destroyAllWindows()
