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

    image = cv.resize(image, (img_rows, img_cols), cv.INTER_NEAREST)
    category = cv.resize(category, (img_rows, img_cols), cv.INTER_NEAREST)

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=[0],
            cval=(0, 255)
        ))
    ])
    seq_det = seq.to_deterministic()

    images = np.zeros((1, img_rows, img_cols, 3), np.uint8)
    images[0] = image
    categories = np.zeros((1, img_rows, img_cols), np.uint8)
    categories[0] = category
    images_aug = seq_det.augment_images(images)
    categories_aug = seq_det.augment_images(categories)
    print(categories_aug[0])
    image = images_aug[0]
    # print(image)
    category_bgr = to_bgr(categories_aug[0])

    cv.imshow('image', image)
    cv.imshow('category_bgr', category_bgr)
    cv.waitKey(0)
    cv.destroyAllWindows()
