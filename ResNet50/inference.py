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

import argparse
import os
import numpy as np
import imagereader


def _inference(img, model):
    pad_x = 0
    pad_y = 0

    if len(img.shape) != 2 and len(img.shape) != 3:
        raise IOError('Invalid number of dimensions for input image. Expecting HW or HWC dimension ordering.')

    if len(img.shape) == 2:
        # add a channel dimension
        img = img.reshape((img.shape[0], img.shape[1], 1))
    img = np.pad(img, pad_width=((0, pad_y), (0, pad_x), (0, 0)), mode='reflect')

    # convert HWC to CHW
    batch_data = img.transpose((2, 0, 1))
    # convert CHW to NCHW
    batch_data = batch_data.reshape((1, batch_data.shape[0], batch_data.shape[1], batch_data.shape[2]))

    pred = model(batch_data)
    pred = np.squeeze(pred)
    return pred


def inference(saved_model_filepath, image_folder, output_folder, image_format):
    # create output filepath
    # if not os.path.exists(output_folder):
    #     os.mkdir(output_folder)

    img_filepath_list = [os.path.join(image_folder, fn) for fn in os.listdir(image_folder) if fn.endswith('.{}'.format(image_format))]

    model = tf.saved_model.load(saved_model_filepath)

    print('Starting inference of file list')
    for i in range(len(img_filepath_list)):
        img_filepath = img_filepath_list[i]
        _, slide_name = os.path.split(img_filepath)
        print('{}/{} : {}'.format(i, len(img_filepath_list), slide_name))

        # print('Loading image: {}'.format(img_filepath))
        img = imagereader.imread(img_filepath)
        img = img.astype(np.float32)

        # normalize with whole image stats
        img = imagereader.zscore_normalize(img)
        # print('  img.shape={}'.format(img.shape))

        pred = _inference(img, model)
        print('  Pred: {} '.format(pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='inference',
                                     description='Script to detect stars with the selected resnet50 classification model')

    parser.add_argument('--saved_model_filepath', dest='saved_model_filepath', type=str,
                        help='SavedModel filepath to the  model to use', required=True)
    parser.add_argument('--image_folder', dest='image_folder', type=str,
                        help='filepath to the folder containing tif images to inference (Required)', required=True)
    parser.add_argument('--output_folder', dest='output_folder', type=str, required=True)
    parser.add_argument('--image_format', dest='image_format', type=str, help='format (extension) of the input images. E.g {tif, jpg, png)', default='tif')

    args = parser.parse_args()

    saved_model_filepath = args.saved_model_filepath
    image_folder = args.image_folder
    output_folder = args.output_folder
    image_format = args.image_format

    print('Arguments:')
    print('saved_model_filepath = {}'.format(saved_model_filepath))
    print('image_folder = {}'.format(image_folder))
    print('output_folder = {}'.format(output_folder))
    print('image_format = {}'.format(image_format))

    inference(saved_model_filepath, image_folder, output_folder, image_format)

