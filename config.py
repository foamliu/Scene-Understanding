import hdf5storage
import numpy as np
import random
import cv2 as cv
import json
import os

img_rows, img_cols = 320, 320
channel = 3
batch_size = 30
epochs = 1000
patience = 50
num_samples = 10335
num_train_samples = 8268
# num_samples - num_train_samples
num_valid_samples = 2067
unknown = 128

num_classes = 37

seg37list = []
for item in hdf5storage.loadmat('seg37list.mat')['seg37list'][0]:
    seg37list.append(item[0])
# print(seg37list)
# ['wall',
#  'floor',
#  'cabinet',
#  'bed',
#  'chair',
#  'sofa',
#  'table',
#  'door',
#  'window',
#  'bookshelf',
#  'picture',
#  'counter',
#  'blinds',
#  'desk',
#  'shelves',
#  'curtain',
#  'dresser',
#  'pillow',
#  'mirror',
#  'floor_mat',
#  'clothes',
#  'ceiling',
#  'books',
#  'fridge',
#  'tv',
#  'paper',
#  'towel',
#  'shower_curtain',
#  'box',
#  'whiteboard',
#  'person',
#  'night_stand',
#  'toilet',
#  'sink',
#  'lamp',
#  'bathtub',
#  'bag']

seg37_dict = dict()
for i in range(len(seg37list)):
    seg37_dict[seg37list[i]] = i


objectColors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd',
                '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
                '#17becf', '#9edae5', '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69',
                '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999', '#621e15', '#e59076', '#128dcd', '#083c52',
                '#64c5f2', '#61afaf', '#0f7369', '#9c9da1', '#365e96', '#983334', '#77973d', '#5d437c', '#36869f',
                '#d1702f', '#8197c5', '#c47f80', '#acc484', '#9887b0', '#2d588a', '#58954c', '#e9a044', '#c12f32',
                '#723e77', '#7d807f', '#9c9ede', '#7375b5', '#4a5584', '#cedb9c', '#b5cf6b', '#8ca252', '#637939',
                '#e7cb94', '#e7ba52', '#bd9e39', '#8c6d31', '#e7969c', '#d6616b', '#ad494a', '#843c39', '#de9ed6',
                '#ce6dbd', '#a55194', '#7b4173', '#000000', '#0000FF']
colors = [[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in objectColors]

folder_2D_segmentation = 'annotation2Dfinal'
folder_rgb_image = 'image'


if __name__ == '__main__':
    filename = '{}_names.txt'.format('train')
    with open(filename, 'r') as f:
        names = f.read().splitlines()

    item = random.choice(names)
    print(item)

    image_path = os.path.join('data', item)
    image_path = os.path.join(image_path, folder_rgb_image)
    image_name = [f for f in os.listdir(image_path) if f.endswith('.jpg')][0]
    image_path = os.path.join(image_path, image_name)
    image = cv.imread(image_path)
    h, w = image.shape[:2]

    seg_path = os.path.join('data', item)
    seg_path = os.path.join(seg_path, folder_2D_segmentation)
    seg_path = os.path.join(seg_path, 'index.json')
    with open(seg_path, 'r') as f:
        seg = json.load(f)

    # print(seg['frames'])
    # print(seg['frames'][0]['polygon'])
    # print(len(seg['frames'][0]['polygon']))

    mask = np.zeros((h, w, 3), np.uint8)

    object_names = []
    for obj in seg['objects']:
        object_names.append(obj['name'])

    for poly in seg['frames'][0]['polygon']:
        object_id = poly['object']
        object_name = object_names[object_id]
        object_color = colors[seg37_dict[object_name]]
        pts = []
        for i in range(len(poly['x'])):
            x = poly['x'][i]
            y = poly['y'][i]
            pts.append([x, y])
        cv.fillPoly(mask, [np.array(pts, np.int32)], object_color)

    cv.imwrite('sample.png', mask)





