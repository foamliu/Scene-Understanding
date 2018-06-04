import os
import random
from random import shuffle

import cv2 as cv
import hdf5storage
import numpy as np
from keras.utils import Sequence

from config import folder_metadata, folder_rgb_image
from config import img_rows, img_cols, batch_size, colors
from config import seg_path


def get_image(name):
    image_path = os.path.join('data', name)
    image_path = os.path.join(image_path, folder_rgb_image)
    image_name = [f for f in os.listdir(image_path) if f.endswith('.jpg')][0]
    image_path = os.path.join(image_path, image_name)
    image = cv.imread(image_path)
    return image


def get_semantic(id):
    filename = os.join(seg_path, '{}.png'.format(id))
    semantic = cv.imread(filename, 0)
    semantic = semantic.astype(np.int32)
    return semantic


def to_bgr(semantic):
    h, w = semantic.shape[:2]
    ret = np.zeros((h, w, 3), np.float32)
    for r in range(h):
        for c in range(w):
            color_id = semantic[r, c]
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
    if crop_size != (img_rows, img_cols):
        ret = cv.resize(ret, dsize=(img_rows, img_cols), interpolation=cv.INTER_NEAREST)
    ret = ret.astype(np.uint8)
    return ret


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        with open('{}_ids.txt'.format(usage), 'r') as f:
            self.ids = f.read().splitlines()

        with open('names.txt', 'r') as f:
            self.names = f.read().splitlines()

        np.random.shuffle(self.ids)

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.ids) - i))
        batch_x = np.empty((length, img_rows, img_cols, 3), dtype=np.float32)
        batch_y = np.empty((length, img_rows, img_cols), dtype=np.int32)

        for i_batch in range(length):
            id = self.ids[i]
            name = self.names[id]
            image = get_image(name)
            semantic = get_semantic(id)
            image_size = image.shape[:2]

            different_sizes = [(320, 320), (480, 480), (640, 640)]
            crop_size = random.choice(different_sizes)

            x, y = random_choice(image_size, crop_size)
            image = safe_crop(image, x, y, crop_size)
            semantic = safe_crop(semantic, x, y, crop_size)

            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                semantic = np.fliplr(semantic)

            x = image / 255.
            y = semantic

            batch_x[i_batch, :, :, 0:3] = x
            batch_y[i_batch, :, :] = y

            i += 1

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
    train_ids = [n for n in range(num_train_samples) if n not in valid_ids]
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
