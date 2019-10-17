import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import skimage.io
import numpy as np
import os
import skimage
import skimage.transform
import argparse
from isg_ai_pb2 import ImageNumberPair
import shutil
import lmdb
import random


def read_image(fp):
    img = skimage.io.imread(fp, as_gray=True)
    return img


def write_img_to_db(txn, img, number, key_str):
    if type(img) is not np.ndarray:
        raise Exception("Img must be numpy array to store into db")
    if type(number) is not np.ndarray:
        number = np.asarray(number)
    if len(img.shape) > 3:
        raise Exception("Img must be 2D or 3D [HW, or HWC] format")
    if len(img.shape) < 2:
        raise Exception("Img must be 2D or 3D [HW, or HWC] format")

    if len(img.shape) == 2:
        # make a 3D array
        img = img.reshape((img.shape[0], img.shape[1], 1))

    datum = ImageNumberPair()
    datum.channels = 1
    datum.img_height = img.shape[0]
    datum.img_width = img.shape[1]
    datum.image = img.tobytes()
    datum.number = number.tobytes()

    datum.img_type = img.dtype.str
    datum.num_type = number.dtype.str

    txn.put(key_str.encode('ascii'), datum.SerializeToString())
    return


def generate_database(img_list, database_name, image_filepath, output_folder):
    output_image_lmdb_file = os.path.join(output_folder, database_name)

    if os.path.exists(output_image_lmdb_file):
        print('Deleting existing database')
        shutil.rmtree(output_image_lmdb_file)

    image_env = lmdb.open(output_image_lmdb_file, map_size=int(5e12))
    image_txn = image_env.begin(write=True)

    for i in range(len(img_list)):
        print('  {}/{}'.format(i, len(img_list)))
        txn_nb = 0
        img_file_name = img_list[i]
        block_key, _ = os.path.splitext(img_file_name)

        img = read_image(os.path.join(image_filepath, img_file_name))
        toks = block_key.split('_')
        num = int(toks[2]) # TODO modify this to make sure classification targets are correct

        key_str = '{}_:{}'.format(block_key, num)
        txn_nb += 1
        write_img_to_db(image_txn, img, num, key_str)

        if txn_nb % 1000 == 0:
            image_txn.commit()
            image_txn = image_env.begin(write=True)

    image_txn.commit()
    image_env.close()


if __name__ == "__main__":
    # Define the inputs
    # ****************************************************

    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='build_lmdb', description='Script which converts folder of images into a pair of lmdb databases for training.')

    parser.add_argument('--image_folder', dest='image_folder', type=str, help='filepath to the folder containing the images', default='../data/images/')
    parser.add_argument('--output_folder', dest='output_folder', type=str, help='filepath to the folder where the outputs will be placed', default='../data/')
    parser.add_argument('--dataset_name', dest='dataset_name', type=str, help='name of the dataset to be used in creating the lmdb files', default='mnist')
    parser.add_argument('--train_fraction', dest='train_fraction', type=float,
                        help='what fraction of the dataset to use for training (0.0, 1.0)', default=0.8)
    parser.add_argument('--image_format', dest='image_format', type=str,
                        help='format (extension) of the input images. E.g {tif, jpg, png)', default='tif')

    args = parser.parse_args()
    image_folder = args.image_folder
    output_folder = args.output_folder
    dataset_name = args.dataset_name
    train_fraction = args.train_fraction
    image_format = args.image_format

    # ****************************************************

    if image_format.startswith('.'):
        # remove leading period
        image_format = image_format[1:]

    image_folder = os.path.abspath(image_folder)
    output_folder = os.path.abspath(output_folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # find the image files for which annotations exist
    img_files = [f for f in os.listdir(image_folder) if f.endswith(image_format)]

    # in place shuffle
    random.shuffle(img_files)

    idx = int(train_fraction * len(img_files))
    train_img_files = img_files[0:idx]
    test_img_files = img_files[idx:]

    print('building train database')
    database_name = 'train-{}.lmdb'.format(dataset_name)
    generate_database(train_img_files, database_name, image_folder, output_folder)

    print('building test database')
    database_name = 'test-{}.lmdb'.format(dataset_name)
    generate_database(test_img_files, database_name, image_folder, output_folder)



