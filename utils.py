import multiprocessing
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
from tensorflow.python.client import device_lib

from config import num_classes, crop_size, folder_rgb_image, seg_path, colors, img_rows, img_cols

prob = np.load('data/prior_prob.npy')
median = np.median(prob)
factor = (median / prob).astype(np.float32)


def categorical_crossentropy_with_class_rebal(y_true, y_pred):
    y_true = K.reshape(y_true, (-1, num_classes))
    y_pred = K.reshape(y_pred, (-1, num_classes))

    idx_max = K.argmax(y_true, axis=1)
    weights = K.gather(factor, idx_max)
    weights = K.reshape(weights, (-1, 1))

    # multiply y_true by weights
    y_true = y_true * weights

    cross_ent = K.categorical_crossentropy(y_pred, y_true)
    cross_ent = K.mean(cross_ent, axis=-1)

    return cross_ent


# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def get_image(name):
    image_path = os.path.join('data', name)
    image_path = os.path.join(image_path, folder_rgb_image)
    image_name = [f for f in os.listdir(image_path) if f.endswith('.jpg')][0]
    image_path = os.path.join(image_path, image_name)
    image = cv.imread(image_path)
    return image


def get_category(id):
    filename = os.path.join(seg_path, '{}.png'.format(id))
    print(filename)
    category = cv.imread(filename, 0)
    return category


def to_bgr(category):
    h, w = category.shape[:2]
    ret = np.zeros((h, w, 3), np.float32)
    for r in range(h):
        for c in range(w):
            color_id = category[r, c]
            # print("color_id: " + str(color_id))
            ret[r, c, :] = colors[color_id]
    ret = ret.astype(np.uint8)
    return ret


def safe_crop(mat, x, y):
    if len(mat.shape) == 2:
        ret = np.zeros((crop_size, crop_size), np.float32)
        interpolation = cv.INTER_NEAREST
    else:
        ret = np.zeros((crop_size, crop_size, 3), np.float32)
        interpolation = cv.INTER_CUBIC
    crop = mat[y:y + crop_size, x:x + crop_size]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    if crop_size != (img_rows, img_cols):
        ret = cv.resize(ret, dsize=(img_rows, img_cols), interpolation=interpolation)
    ret = ret.astype(np.uint8)
    return ret


def random_crop(image, category):
    height, width = image.shape[:2]
    x = random.randint(0, max(0, width - crop_size))
    y = random.randint(0, max(0, height - crop_size))
    image = safe_crop(image, x, y)
    category = safe_crop(category, x, y)
    return image, category
