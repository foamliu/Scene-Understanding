import random
import unittest

import cv2 as cv
import numpy as np
import os

from data_generator import get_image, get_semantic, to_bgr
from data_generator import random_choice, safe_crop

class TestStringMethods(unittest.TestCase):
    def setUp(self):
        if not os.path.exists('temp'):
            os.makedirs('temp')

    def tearDown(self):
        pass

    def test_get_semantic(self):
        name = 'SUNRGBD/kv1/NYUdata/NYU0899'
        image, image_size = get_image(name)
        print('image_size: ' + str(image_size))
        semantic = get_semantic(name, image_size)
        semantic = to_bgr(semantic)
        cv.imwrite('temp/test_get_semantic_image.png', image)
        cv.imwrite('temp/test_get_semantic_semantic.png', semantic)

    def test_safe_crop(self):
        name = 'SUNRGBD/kv1/NYUdata/NYU0899'
        image, image_size = get_image(name)
        semantic = get_semantic(name, image_size)
        different_sizes = [(320, 320), (480, 480), (640, 640)]
        crop_size = random.choice(different_sizes)

        x, y = random_choice(image_size, crop_size)
        image = safe_crop(image, x, y, crop_size)
        semantic = safe_crop(semantic, x, y, crop_size)
        semantic = to_bgr(semantic)
        cv.imwrite('temp/test_safe_crop_image.png', image)
        cv.imwrite('temp/test_safe_crop_semantic.png', semantic)

    #
    # def test_different_sizes(self):
    #     different_sizes = [(320, 320), (320, 320), (320, 320), (480, 480), (640, 640)]
    #     crop_size = random.choice(different_sizes)
    #     # print('crop_size=' + str(crop_size))
    #
    # def test_resize(self):
    #     name = '0_0.png'
    #     filename = os.path.join('merged', name)
    #     image = cv.imread(filename)
    #     bg_h, bg_w = image.shape[:2]
    #     a = get_alpha(name)
    #     a_h, a_w = a.shape[:2]
    #     alpha = np.zeros((bg_h, bg_w), np.float32)
    #     alpha[0:a_h, 0:a_w] = a
    #     trimap = generate_trimap(alpha)
    #     # 剪切尺寸 320:640:480 = 3:1:1
    #     crop_size = (480, 480)
    #     x, y = random_choice(trimap, crop_size)
    #     image = safe_crop(image, x, y, crop_size)
    #     trimap = safe_crop(trimap, x, y, crop_size)
    #     alpha = safe_crop(alpha, x, y, crop_size)
    #     cv.imwrite('temp/test_resize_image.png', image)
    #     cv.imwrite('temp/test_resize_trimap.png', trimap)
    #     cv.imwrite('temp/test_resize_alpha.png', alpha)


if __name__ == '__main__':
    unittest.main()
