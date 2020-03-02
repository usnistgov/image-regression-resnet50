# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 Required')

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    raise RuntimeError('Tensorflow 2.x.x required')

import multiprocessing
from multiprocessing import Process
import queue
import random
import traceback
import lmdb
import numpy as np
import augment
import os
import skimage.io
import skimage.transform
from isg_ai_pb2 import ImageNumberPair


def zscore_normalize(image_data):
    image_data = image_data.astype(np.float32)

    if len(image_data.shape) == 3:
        # input is CHW
        for c in range(image_data.shape[0]):
            std = np.std(image_data[c, :, :])
            mv = np.mean(image_data[c, :, :])
            if std <= 1.0:
                # normalize (but dont divide by zero)
                image_data[c, :, :] = (image_data[c, :, :] - mv)
            else:
                # z-score normalize
                image_data[c, :, :] = (image_data[c, :, :] - mv) / std
    elif len(image_data.shape) == 2:
        # input is HW
        std = np.std(image_data)
        mv = np.mean(image_data)
        if std <= 1.0:
            # normalize (but dont divide by zero)
            image_data = (image_data - mv)
        else:
            # z-score normalize
            image_data = (image_data - mv) / std
    else:
        raise IOError("Input to Z-Score normalization needs to be either a 2D or 3D image [HW, or CHW]")

    return image_data


def imread(fp):
    return skimage.io.imread(fp)


def imwrite(img, fp):
    skimage.io.imsave(fp, img)


class ImageReader:
    # setup the image data augmentation parameters
    _reflection_flag = True
    _rotation_flag = True
    _jitter_augmentation_severity = 0.1  # x% of a FOV
    _noise_augmentation_severity = 0.02  # vary noise by x% of the dynamic range present in the image
    _scale_augmentation_severity = 0.1  # vary size by x%
    _blur_max_sigma = 2  # pixels
    _intensity_augmentation_severity = None  # vary intensity by x% of the dynamic range present in the image

    def __init__(self, img_db, use_augmentation=True, shuffle=True, num_workers=1):
        random.seed()

        # copy inputs to class variables
        self.image_db = img_db
        self.use_augmentation = use_augmentation
        self.shuffle = shuffle
        self.nb_workers = num_workers

        # init class state
        self.queue_starvation = False
        self.maxOutQSize = num_workers * 100  # queue 100 images per reader
        self.workers = None
        self.done = False

        # setup queue mechanism
        self.terminateQ = multiprocessing.Queue(maxsize=self.nb_workers)  # limit output queue size
        self.outQ = multiprocessing.Queue(maxsize=self.maxOutQSize)  # limit output queue size
        self.idQ = multiprocessing.Queue(maxsize=self.nb_workers)

        # confirm that the input database exists
        if not os.path.exists(self.image_db):
            print('Could not load database file: ')
            print(self.image_db)
            raise IOError("Missing Database")

        # get a list of keys from the lmdb
        self.keys_flat = list()
        self.keys = list()
        self.keys.append(list())  # there will always be at least one class

        self.lmdb_env = lmdb.open(self.image_db, map_size=int(2e10), readonly=True)  # 20 GB
        self.lmdb_txns = list()

        datum = ImageNumberPair()  # create a datum for decoding serialized protobuf objects
        print('Initializing image database')

        with self.lmdb_env.begin(write=False) as lmdb_txn:
            cursor = lmdb_txn.cursor()

            # move cursor to the first element
            cursor.first()
            # get the first serialized value from the database and convert from serialized representation
            datum.ParseFromString(cursor.value())
            # record the image size
            self.image_size = [datum.img_height, datum.img_width, datum.channels]

            cursor = lmdb_txn.cursor().iternext(keys=True, values=False)
            # iterate over the database getting the keys
            for key in cursor:
                self.keys_flat.append(key)

        print('Dataset has {} examples'.format(len(self.keys_flat)))

    def get_image_count(self):
        # tie epoch size to the number of images
        return int(len(self.keys_flat))

    def get_image_size(self):
        return self.image_size

    def get_image_tensor_shape(self):
        # HWC to CHW
        return [self.image_size[2], self.image_size[0], self.image_size[1]]

    def get_label_tensor_shape(self):
        return [self.image_size[0], self.image_size[1]]

    def startup(self):
        self.workers = None
        self.done = False

        [self.idQ.put(i) for i in range(self.nb_workers)]
        [self.lmdb_txns.append(self.lmdb_env.begin(write=False)) for i in range(self.nb_workers)]
        # launch workers
        self.workers = [Process(target=self.__image_loader) for i in range(self.nb_workers)]

        # start workers
        for w in self.workers:
            w.start()

    def shutdown(self):
        # tell workers to shutdown
        for w in self.workers:
            self.terminateQ.put(None)

        # empty the output queue (to allow blocking workers to terminate
        nb_none_received = 0
        # empty output queue
        while nb_none_received < len(self.workers):
            try:
                while True:
                    val = self.outQ.get_nowait()
                    if val is None:
                        nb_none_received += 1
            except queue.Empty:
                pass  # do nothing

        # wait for the workers to terminate
        for w in self.workers:
            w.join()

    def __get_next_key(self):
        if self.shuffle:
            # select a key at random from the list (does not account for class imbalance)
            fn = self.keys_flat[random.randint(0, len(self.keys_flat) - 1)]
        else:  # no shuffle
            # without shuffle you cannot balance classes
            fn = self.keys_flat[self.key_idx]
            self.key_idx += self.nb_workers
            self.key_idx = self.key_idx % len(self.keys_flat)

        return fn

    def __image_loader(self):
        termimation_flag = False  # flag to control the worker shutdown
        self.key_idx = self.idQ.get()  # setup non-shuffle index to stride across flat keys properly
        try:
            datum = ImageNumberPair()  # create a datum for decoding serialized objects

            local_lmdb_txn = self.lmdb_txns[self.key_idx]

            # while the worker has not been told to terminate, loop infinitely
            while not termimation_flag:

                # poll termination queue for shutdown command
                try:
                    if self.terminateQ.get_nowait() is None:
                        termimation_flag = True
                        break
                except queue.Empty:
                    pass  # do nothing

                # build a single image selecting the labels using round robin through the shuffled order

                fn = self.__get_next_key()

                # extract the serialized image from the database
                value = local_lmdb_txn.get(fn)
                # convert from serialized representation
                datum.ParseFromString(value)

                # convert from string to numpy array
                I = np.fromstring(datum.image, dtype=datum.img_type)
                # reshape the numpy array using the dimensions recorded in the datum
                I = I.reshape((datum.img_height, datum.img_width, datum.channels))

                number = np.fromstring(datum.number, dtype=datum.num_type).reshape(-1)

                if self.use_augmentation:
                    I = I.astype(np.float32)

                    I = augment.augment_image(I,
                                                 reflection_flag=self._reflection_flag,
                                                 rotation_flag=self._rotation_flag,
                                                 jitter_augmentation_severity=self._jitter_augmentation_severity,
                                                 noise_augmentation_severity=self._noise_augmentation_severity,
                                                 scale_augmentation_severity=self._scale_augmentation_severity,
                                                 blur_augmentation_max_sigma=self._blur_max_sigma,
                                                 intensity_augmentation_severity=self._intensity_augmentation_severity)

                # format the image into a tensor
                # reshape into tensor (CHW)
                I = I.transpose((2, 0, 1))
                I = I.astype(np.float32)
                I = zscore_normalize(I)

                number = number.astype(np.float32)

                # add the batch in the output queue
                # this put block until there is space in the output queue (size 50)
                self.outQ.put((I, number))

        except Exception as e:
            print('***************** Reader Error *****************')
            print(e)
            traceback.print_exc()
            print('***************** Reader Error *****************')
        finally:
            # when the worker terminates add a none to the output so the parent gets a shutdown confirmation from each worker
            self.outQ.put(None)

    def get_example(self):
        # get a ready to train batch from the output queue and pass to to the caller
        if self.outQ.qsize() < int(0.1 * self.maxOutQSize):
            if not self.queue_starvation:
                print('Input Queue Starvation !!!!')
            self.queue_starvation = True
        if self.queue_starvation and self.outQ.qsize() > int(0.5 * self.maxOutQSize):
            print('Input Queue Starvation Over')
            self.queue_starvation = False
        return self.outQ.get()

    def generator(self):
        while True:
            batch = self.get_example()
            if batch is None:
                return
            yield batch

    def get_queue_size(self):
        return self.outQ.qsize()

    def get_tf_dataset(self):
        print('Creating Dataset')
        # wrap the input queues into a Dataset
        # this sets up the imagereader class as a Python generator
        # Images come in as HWC, and are converted into CHW for network
        image_shape = tf.TensorShape((self.image_size[2], self.image_size[0], self.image_size[1]))
        label_shape = tf.TensorShape((1))
        return tf.data.Dataset.from_generator(self.generator, output_types=(tf.float32, tf.float32),
                                              output_shapes=(image_shape, label_shape))


