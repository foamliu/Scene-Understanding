import random
import unittest

import cv2 as cv
import numpy as np
import os

from data_generator import get_image, get_semantic, to_bgr


class TestStringMethods(unittest.TestCase):

    def test_get_semantic(self):
        name = 'SUNRGBD/kv2/kinect2data/003489_2014-05-21_15-49-49_094959634447_rgbf000101-resize'
        image, image_size = get_image(name)
        print('image_size: ' + str(image_size))
        semantic = get_semantic(name, image_size)
        semantic = to_bgr(semantic)
        cv.imwrite('temp/test_get_semantic_image.png', image)
        cv.imwrite('temp/test_get_semantic_semantic.png', semantic)

    # def test_flip(self):
    #     image = cv.imread('fg/1-1252426161dfXY.jpg')
    #     # print(image.shape)
    #     alpha = cv.imread('mask/1-1252426161dfXY.jpg', 0)
    #     trimap = generate_trimap(alpha)
    #     x, y = random_choice(trimap)
    #     image = safe_crop(image, x, y)
    #     trimap = safe_crop(trimap, x, y)
    #     alpha = safe_crop(alpha, x, y)
    #     image = np.fliplr(image)
    #     trimap = np.fliplr(trimap)
    #     alpha = np.fliplr(alpha)
    #     cv.imwrite('temp/test_flip_image.png', image)
    #     cv.imwrite('temp/test_flip_trimap.png', trimap)
    #     cv.imwrite('temp/test_flip_alpha.png', alpha)
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
