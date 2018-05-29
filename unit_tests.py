import os
import random
import unittest

import cv2 as cv

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


if __name__ == '__main__':
    unittest.main()
