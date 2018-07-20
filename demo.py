# import the necessary packages
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
from keras.applications.vgg16 import preprocess_input

from config import img_rows, img_cols, num_classes
from model import build_model
from utils import get_image, get_category, random_crop, to_bgr

if __name__ == '__main__':
    model_weights_path = 'models/model.81-3.5244.hdf5'
    model = build_model()
    model.load_weights(model_weights_path)

    print(model.summary())

    with open('names.txt', 'r') as f:
        names = f.read().splitlines()

    filename = 'valid_ids.txt'
    with open(filename, 'r') as f:
        ids = f.read().splitlines()
        ids = list(map(int, ids))
    samples = random.sample(ids, 10)
    print('samples: ' + str(samples))

    for i in range(len(samples)):
        id = samples[i]
        name = names[id]
        image_bgr = get_image(name)
        category = get_category(id)
        image_bgr, category = random_crop(image_bgr, category)

        image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        colorful_category = to_bgr(category)
        print('Start processing image: {}'.format(name))

        x_test = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
        x_test[0, :, :, 0:3] = image_rgb
        x_test = preprocess_input(x_test)

        out = model.predict(x_test)
        out = np.reshape(out, (img_rows, img_cols, num_classes))
        out = np.argmax(out, axis=2)
        out = to_bgr(out)

        if not os.path.exists('images'):
            os.makedirs('images')

        cv.imwrite('images/{}_image.png'.format(i), image_bgr)
        cv.imwrite('images/{}_out.png'.format(i), out)
        cv.imwrite('images/{}_gt.png'.format(i), colorful_category)

    K.clear_session()
