import os
import random
from random import shuffle

import cv2 as cv
import numpy as np
from keras.utils import Sequence

from config import batch_size
from config import colors
from config import img_cols
from config import img_rows
from config import num_classes

train_folder = 'data/rgb'
depth_folder = 'data/depth'
semantic_folder = 'data/semantic'


def get_semantic(name):
    tokens = name.split('_')
    tokens[-1] = 'semantic_pretty.png'
    name = '_'.join(tokens)
    filename = os.path.join(semantic_folder, name)
    semantic = cv.imread(filename)
    return semantic


def get_y(semantic):
    temp = np.zeros(shape=(320, 320, num_classes), dtype=np.int32)
    semantic = np.array(semantic).astype(np.int32)
    for i in range(num_classes):
        temp[:, :, i] = np.sum(np.abs(semantic - colors[i]), axis=2)
    y = np.argmin(temp, axis=2)
    return y


def to_bgr(y_pred):
    ret = np.zeros((img_rows, img_cols, 3), np.float32)
    for r in range(320):
        for c in range(320):
            color_id = y_pred[r, c]
            # print("color_id: " + str(color_id))
            ret[r, c, :] = colors[color_id]
    ret = ret.astype(np.uint8)
    return ret


def random_choice(image_size, crop_size):
    height, width = image_size
    crop_height, crop_width = crop_size
    x = random.randint(0, max(0, width - crop_width))
    y = random.randint(0, max(0, height - crop_height))
    return x, y


def safe_crop(mat, x, y, crop_size):
    crop_height, crop_width = crop_size
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.float32)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.float32)
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    if crop_size != (320, 320):
        ret = cv.resize(ret, dsize=(img_rows, img_cols), interpolation=cv.INTER_CUBIC)
    return ret


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        filename = '{}_names.txt'.format(usage)
        with open(filename, 'r') as f:
            self.names = f.read().splitlines()

        np.random.shuffle(self.names)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_rows, img_cols, 3), dtype=np.float32)
        batch_y = np.empty((length, img_rows, img_cols), dtype=np.int32)

        for i_batch in range(length):
            name = self.names[i]
            filename = os.path.join(train_folder, name)
            image = cv.imread(filename)
            image_size = image.shape[:2]
            semantic = get_semantic(name)

            different_sizes = [(320, 320), (480, 480), (480, 480), (480, 480), (640, 640), (640, 640), (640, 640),
                               (960, 960), (960, 960), (960, 960)]
            crop_size = random.choice(different_sizes)

            x, y = random_choice(image_size, crop_size)
            image = safe_crop(image, x, y, crop_size)
            semantic = safe_crop(semantic, x, y, crop_size)

            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                semantic = np.fliplr(semantic)

            x = image / 255.
            y = get_y(semantic)

            batch_x[i_batch, :, :, 0:3] = x
            batch_y[i_batch, :, :] = y

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')


def split_data():
    train_folder = 'data/rgb'
    names = [f for f in os.listdir(train_folder) if f.endswith('.png')]
    num_samples = len(names)  # 52903
    print('num_samples: ' + str(num_samples))

    num_train_samples = int(num_samples * 0.8)
    print('num_train_samples: ' + str(num_train_samples))
    num_valid_samples = num_samples - num_train_samples
    print('num_valid_samples: ' + str(num_valid_samples))
    valid_names = random.sample(names, num_valid_samples)
    train_names = [n for n in names if n not in valid_names]
    shuffle(valid_names)
    shuffle(train_names)

    with open('valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))


if __name__ == '__main__':
    split_data()
