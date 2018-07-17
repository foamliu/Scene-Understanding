import os
import random
from random import shuffle

import cv2 as cv
import hdf5storage
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.utils import Sequence
from keras.utils import to_categorical

from config import folder_metadata, folder_rgb_image
from config import img_rows, img_cols, batch_size, colors
from config import seg_path, num_classes, crop_size


def get_image(name):
    image_path = os.path.join('data', name)
    image_path = os.path.join(image_path, folder_rgb_image)
    image_name = [f for f in os.listdir(image_path) if f.endswith('.jpg')][0]
    image_path = os.path.join(image_path, image_name)
    image = cv.imread(image_path)
    return image


def get_category(id):
    filename = os.path.join(seg_path, '{}.png'.format(id))
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


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        with open('{}_ids.txt'.format(usage), 'r') as f:
            ids = f.read().splitlines()
            self.ids = list(map(int, ids))

        with open('names.txt', 'r') as f:
            self.names = f.read().splitlines()

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.ids) - i))
        batch_x = np.empty((length, img_rows, img_cols, 3), dtype=np.float32)
        batch_y = np.empty((length, img_rows, img_cols, num_classes), dtype=np.int32)

        for i_batch in range(length):
            id = self.ids[i + i_batch]
            name = self.names[id]
            image = get_image(name)
            category = get_category(id)
            image, category = random_crop(image, category)

            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                category = np.fliplr(category)

            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            batch_x[i_batch, :, :, 0:3] = image
            batch_y[i_batch, :, :] = to_categorical(category, num_classes)

        batch_x = preprocess_input(batch_x)

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.ids)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')


def split_data():
    filename = os.path.join(folder_metadata, 'SUNRGBDMeta.mat')
    meta = hdf5storage.loadmat(filename)
    names = []
    for item in meta['SUNRGBDMeta'][0]:
        name = item[0][0]
        names.append(name)

    num_samples = len(names)  # 10335
    print('num_samples: ' + str(num_samples))

    num_train_samples = int(num_samples * 0.8)
    print('num_train_samples: ' + str(num_train_samples))
    num_valid_samples = num_samples - num_train_samples
    print('num_valid_samples: ' + str(num_valid_samples))
    valid_ids = random.sample(range(num_train_samples), num_valid_samples)
    valid_ids = list(map(str, valid_ids))
    train_ids = [str(n) for n in range(num_train_samples) if n not in valid_ids]
    shuffle(valid_ids)
    shuffle(train_ids)

    with open('names.txt', 'w') as file:
        file.write('\n'.join(names))

    with open('valid_ids.txt', 'w') as file:
        file.write('\n'.join(valid_ids))

    with open('train_ids.txt', 'w') as file:
        file.write('\n'.join(train_ids))


if __name__ == '__main__':
    split_data()
