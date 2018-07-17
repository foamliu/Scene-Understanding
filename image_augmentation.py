import random

import cv2 as cv

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
    category_bgr = to_bgr(category)
    image = cv.resize(image, (img_rows, img_cols), cv.INTER_CUBIC)
    category = cv.resize(category, (img_rows, img_cols), cv.INTER_NEAREST)

    cv.imshow('image', image)
    cv.imshow('category_bgr', category_bgr)
    cv.waitKey(0)
    cv.destroyAllWindows()
