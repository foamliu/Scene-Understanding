import os
import random
from random import shuffle

import cv2 as cv
import hdf5storage
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.utils import Sequence
from keras.utils import to_categorical

from config import folder_metadata
from config import img_rows, img_cols, batch_size
from config import num_classes
from image_augmentation import seq_det, seq_img
from utils import get_image, get_category


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
        X = np.empty((length, img_rows, img_cols, 3), dtype=np.float32)
        batch_y = np.empty((length, img_rows, img_cols), dtype=np.uint8)
        Y = np.empty((length, img_rows, img_cols, num_classes), dtype=np.float32)

        for i_batch in range(length):
            id = self.ids[i + i_batch]
            name = self.names[id]
            image = get_image(name)
            category = get_category(id)
            image = cv.resize(image, (img_rows, img_cols), cv.INTER_NEAREST)
            category = cv.resize(category, (img_rows, img_cols), cv.INTER_NEAREST)

            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            X[i_batch] = image
            batch_y[i_batch] = category

        X = seq_img.augment_images(X)
        X = seq_det.augment_images(X)
        X = preprocess_input(X)

        batch_y = seq_det.augment_images(batch_y)

        for i_batch in range(length):
            Y[i_batch] = to_categorical(batch_y[i_batch], num_classes)

        return X, Y

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
