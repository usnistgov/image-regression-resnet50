# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 Required')

import numpy as np
import skimage.io
import skimage.transform
import os


def convert(img_filepath, label_filepath, output_dirpath, output_csv_filepath, n):
    f = open(img_filepath, "rb")
    l = open(label_filepath, "rb")
    if not os.path.exists(output_dirpath):
        os.mkdir(output_dirpath)

    f.read(16)
    l.read(8)

    with open(output_csv_filepath, 'w') as fh:
        for i in range(n):
            image = []
            for j in range(28*28):
                image.append(ord(f.read(1)))
            img = np.asarray(image, dtype=np.uint8)
            img = img.reshape((28,28))
            img = img.astype(np.uint8)

            class_id = int(ord(l.read(1)))
            img_fn = 'img_{:08d}_{}.tif'.format(i, class_id)
            skimage.io.imsave(os.path.join(output_dirpath, img_fn), img, compress=6)
            fh.write('{}, {}\n'.format(img_fn, class_id))
    f.close()
    l.close()


# Modify this filepath to point to the folder where you downloaded MNIST from http://yann.lecun.com/exdb/mnist/
fp = '/scratch/datasets/MNIST/'

img_filepath = os.path.join(fp, 'train-images-idx3-ubyte')
label_filepath = os.path.join(fp, 'train-labels-idx1-ubyte')
output_dirpath = os.path.join(fp, 'train')
output_csv_filepath = os.path.join(fp, 'train_ground_truth.csv')
convert(img_filepath, label_filepath, output_dirpath, output_csv_filepath, 60000)

img_filepath = os.path.join(fp, 't10k-images-idx3-ubyte')
label_filepath = os.path.join(fp, 't10k-labels-idx1-ubyte')
output_dirpath = os.path.join(fp, 'test')
output_csv_filepath = os.path.join(fp, 'teat_ground_truth.csv')
convert(img_filepath, label_filepath, output_dirpath, output_csv_filepath, 10000)
