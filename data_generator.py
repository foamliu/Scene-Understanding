import json
import os
import random
from random import shuffle

import cv2 as cv
import hdf5storage
import numpy as np
from keras.utils import Sequence

from config import folder_metadata, folder_rgb_image, folder_2D_segmentation
from config import img_rows, img_cols, batch_size, colors
from config import seg38_dict


def get_image(name):
    image_path = os.path.join('data', name)
    image_path = os.path.join(image_path, folder_rgb_image)
    image_name = [f for f in os.listdir(image_path) if f.endswith('.jpg')][0]
    image_path = os.path.join(image_path, image_name)
    image = cv.imread(image_path)
    image_size = image.shape[:2]
    return image, image_size


def get_semantic(name, image_size):
    seg_path = os.path.join('data', name)
    seg_path = os.path.join(seg_path, folder_2D_segmentation)
    seg_path = os.path.join(seg_path, 'index.json')

    h, w = image_size
    semantic = np.zeros((h, w, 1), np.int32)

    with open(seg_path, 'r') as f:
        seg = json.load(f)

    object_names = []
    for obj in seg['objects']:
        if not obj:
            object_names.append(None)
        else:
            object_names.append(obj['name'])

    for poly in seg['frames'][0]['polygon']:
        object_id = poly['object']
        object_name = object_names[object_id]
        if object_name in seg38_dict.keys():
            class_id = (seg38_dict[object_name])
            pts = []
            for i in range(len(poly['x'])):
                x = poly['x'][i]
                y = poly['y'][i]
                pts.append([x, y])
                cv.fillPoly(semantic, [np.array(pts, np.int32)], class_id)

    semantic = np.reshape(semantic, (h, w))
    return semantic


def to_bgr(y_pred):
    h, w = y_pred.shape[:2]
    ret = np.zeros((h, w, 3), np.float32)
    for r in range(h):
        for c in range(w):
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
            image, image_size = get_image(name)
            semantic = get_semantic(name, image_size)

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
        np.random.shuffle(self.names)


def is_valid(name):
    seg_path = os.path.join('data', name)
    seg_path = os.path.join(seg_path, folder_2D_segmentation)
    seg_path = os.path.join(seg_path, 'index.json')
    try:
        with open(seg_path, 'r') as f:
            seg = json.load(f)

        object_names = []
        for obj in seg['objects']:
            if not obj:
                object_names.append(None)
            else:
                object_names.append(obj['name'])

        for poly in seg['frames'][0]['polygon']:
            object_id = poly['object']
            if object_id >= len(object_names):
                return False
            if not isinstance(poly['x'], list) or not isinstance(poly['y'], list):
                return False
            object_name = object_names[object_id]
            if object_name in seg38_dict.keys():
                # class_id = seg38_dict[object_name]
                pts = []
                for i in range(len(poly['x'])):
                    x = poly['x'][i]
                    y = poly['y'][i]
                    pts.append([x, y])
                if len(pts) < 3:
                    return False

    except json.decoder.JSONDecodeError:
        return False

    return True


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
        if is_valid(name):
            names.append(item[0][0])

    num_samples = len(names)  # 10335
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
