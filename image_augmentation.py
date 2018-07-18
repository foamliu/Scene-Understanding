import random

import cv2 as cv
import numpy as np
from imgaug import augmenters as iaa
from tqdm import tqdm

from config import img_rows, img_cols
from data_generator import get_image, get_category, to_bgr

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.CropAndPad(
        percent=(-0.25, 0.25),
        pad_mode=["wrap"],
    ),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8),
        order=[0],
        mode='wrap'
    )
])
seq_det = seq.to_deterministic()

seq_img = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0, 0.5)),
    iaa.ContrastNormalization((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),

], random_order=True)

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

    length = 10
    images = np.zeros((length, img_rows, img_cols, 3), np.uint8)
    categories = np.zeros((length, img_rows, img_cols), np.uint8)
    for i in tqdm(range(length)):
        images[i] = image.copy()
        categories[i] = category.copy()

    images_aug = seq_img.augment_images(images)
    images_aug = seq_det.augment_images(images_aug)
    categories_aug = seq_det.augment_images(categories)

    for i in range(length):
        image = images_aug[i]
        category_bgr = to_bgr(categories_aug[i].astype(np.uint8))
        cv.imwrite('images/{}_image_aug.png'.format(i), image)
        cv.imwrite('images/{}_category_aug.png'.format(i), category_bgr)
