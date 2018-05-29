# -*- coding: utf-8 -*-
import argparse
import os
import zipfile

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
