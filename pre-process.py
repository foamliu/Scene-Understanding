# -*- coding: utf-8 -*-
import argparse
import os
import zipfile

import cv2 as cv
import hdf5storage
import numpy as np
from console_progressbar import ProgressBar

# python pre-process.py -d ../../data/Semantic-Segmentation/data/
if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", help="path to data files")
    args = vars(ap.parse_args())
    data_path = args["data"]

    if data_path is None:
        data_path = 'data/'

    filename = 'SUNRGBD.zip'
    filename = os.path.join(data_path, filename)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_file:
        zip_file.extractall(data_path)

    filename = 'SUNRGBDtoolbox.zip'
    filename = os.path.join(data_path, filename)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_file:
        zip_file.extractall(data_path)

    filename = 'data/SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat'
    SUNRGBD2Dseg = hdf5storage.loadmat(filename)
    num_samples = len(SUNRGBD2Dseg['SUNRGBD2Dseg'][0])

    seg_path = 'data/SUNRGBD2Dseg'
    if not os.path.exists(seg_path):
        os.makedirs(seg_path)

    pb = ProgressBar(total=num_samples, prefix='Processing images', suffix='', decimals=3, length=50, fill='=')

    for i in range(num_samples):
        semantic = SUNRGBD2Dseg[0][i][0]
        semantic = semantic.astype(np.uint8)
        filename = os.join(seg_path, '{}.png'.format(i))
        cv.imwrite(filename, semantic)
        pb.print_progress_bar(i + 1)
