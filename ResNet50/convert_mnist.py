# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import numpy as np
import skimage.io
import skimage.transform
import os

# http://yann.lecun.com/exdb/mnist/

def convert(imgf, labelf, ofp, n):
    f = open(imgf, "rb")
    l = open(labelf, "rb")
    if not os.path.exists(ofp):
        os.mkdir(ofp)

    f.read(16)
    l.read(8)

    for i in range(n):
        image = []
        for j in range(28*28):
            image.append(ord(f.read(1)))
        img = np.asarray(image, dtype=np.uint8)
        img = img.reshape((28,28))
        # img = skimage.transform.resize(img, (28 * 2, 28 * 2), preserve_range=True)
        img = skimage.transform.rescale(img, (4, 4), preserve_range=True, multichannel=False)
        img = img.astype(np.uint8)

        skimage.io.imsave(os.path.join(ofp, 'img_{:08d}_{}.tif'.format(i, int(ord(l.read(1))))), img)
    f.close()
    l.close()

# convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "mnist_train", 60000)
# convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", "mnist_test", 10000)
# convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "mnist_train", 1000)
convert("/home/mmajursk/Downloads/mnist/t10k-images-idx3-ubyte", "/home/mmajursk/Downloads/mnist/t10k-labels-idx1-ubyte", "/home/mmajursk/Gitlab/Regression/data/images", 200)
