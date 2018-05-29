# import the necessary packages
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from config import num_classes
from data_generator import random_choice, safe_crop, to_bgr
from model import build_encoder_decoder
from utils import draw_str


def get_semantic(name):
    label_test_path = 'data/semantic_test/'
    tokens = name.split('_')
    tokens[-1] = 'semantic_pretty.png'
    name = '_'.join(tokens)
    filename = os.path.join(label_test_path, name)
    label = cv.imread(filename)
    return label


if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    channel = 3

    model_weights_path = 'models/model.64-2.1187.hdf5'
    model = build_encoder_decoder()
    model.load_weights(model_weights_path)

    print(model.summary())

    rgb_test_path = 'data/rgb_test/'
    label_test_path = 'data/semantic_test/'
    test_images = [f for f in os.listdir(rgb_test_path) if
                   os.path.isfile(os.path.join(rgb_test_path, f)) and f.endswith('.png')]

    samples = random.sample(test_images, 10)

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(rgb_test_path, image_name)
        image = cv.imread(filename)
        label = get_semantic(image_name)
        image_size = image.shape[:2]
        different_sizes = [(320, 320), (480, 480), (640, 640)]
        crop_size = random.choice(different_sizes)

        x, y = random_choice(image_size, crop_size)
        image = safe_crop(image, x, y, crop_size)
        label = safe_crop(label, x, y, crop_size)
        print('Start processing image: {}'.format(filename))

        x_test = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
        x_test[0, :, :, 0:3] = image / 255.

        out = model.predict(x_test)
        out = np.reshape(out, (img_rows, img_cols, num_classes))
        out = np.argmax(out, axis=2)
        out = to_bgr(out)

        str_msg = 'crop_size: %s' % (str(crop_size))
        draw_str(out, (20, 20), str_msg)

        if not os.path.exists('images'):
            os.makedirs('images')

        cv.imwrite('images/{}_image.png'.format(i), image)
        cv.imwrite('images/{}_out.png'.format(i), out)
        cv.imwrite('images/{}_label.png'.format(i), label)

    K.clear_session()
