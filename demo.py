# import the necessary packages
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from config import num_classes
from data_generator import get_image, get_category
from data_generator import to_bgr
from model import build_encoder_decoder

if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    channel = 3

    model_weights_path = 'models/model.16-3.1407.hdf5'
    model = build_encoder_decoder()
    model.load_weights(model_weights_path)

    print(model.summary())

    filename = '{}_names.txt'.format('valid')
    with open(filename, 'r') as f:
        names = f.read().splitlines()
    samples = random.sample(names, 10)

    for i in range(len(samples)):
        name = samples[i]
        image, image_size = get_image(name)
        category = get_category(name)
        colorful_category = to_bgr(category)
        print('Start processing image: {}'.format(name))

        x_test = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
        x_test[0, :, :, 0:3] = image / 255.

        out = model.predict(x_test)
        out = np.reshape(out, (img_rows, img_cols, num_classes))
        out = np.argmax(out, axis=2)
        out = to_bgr(out)

        if not os.path.exists('images'):
            os.makedirs('images')

        cv.imwrite('images/{}_image.png'.format(i), image)
        cv.imwrite('images/{}_out.png'.format(i), out)
        cv.imwrite('images/{}_semantic.png'.format(i), colorful_category)

    K.clear_session()
