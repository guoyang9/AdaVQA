import os
import csv
import sys
import base64
import argparse
sys.path.append(os.getcwd())
csv.field_size_limit(sys.maxsize)

import h5py
import numpy as np
from tqdm import tqdm

import utils.config as config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

    features_shape = (
        82783 + 40504 if not args.test else 81434,  # number of images in trainval or in test
        config.output_features,
        config.rcnn_output_size,
    )
    boxes_shape = (
        features_shape[0],
        4,
        config.rcnn_output_size,
    )

    if not args.test:
        path = config.rcnn_trainval_path
    else:
        path = config.rcnn_test_path
    with h5py.File(path, 'w', libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float32')
        boxes = fd.create_dataset('boxes', shape=boxes_shape, dtype='float32')
        coco_ids = fd.create_dataset('ids', shape=(features_shape[0],), dtype='int32')
        widths = fd.create_dataset('widths', shape=(features_shape[0],), dtype='int32')
        heights = fd.create_dataset('heights', shape=(features_shape[0],), dtype='int32')

        if not args.test:
            path = config.bottom_up_trainval_path
        else:
            path = config.bottom_up_test_path
        fd = open(path, 'r')
        reader = csv.DictReader(fd, delimiter='\t', fieldnames=FIELDNAMES)

        for i, item in enumerate(tqdm(reader, total=features_shape[0])):
            coco_ids[i] = int(item['image_id'])
            widths[i] = int(item['image_w'])
            heights[i] = int(item['image_h'])

            buf = base64.decodebytes(item['features'].encode('utf8'))
            array = np.frombuffer(buf, dtype='float32')
            array = array.reshape((-1, config.output_features)).transpose()
            features[i, :, :array.shape[1]] = array

            buf = base64.decodebytes(item['boxes'].encode('utf8'))
            array = np.frombuffer(buf, dtype='float32')
            array = array.reshape((-1, 4)).transpose()
            boxes[i, :, :array.shape[1]] = array


if __name__ == '__main__':
    main()
