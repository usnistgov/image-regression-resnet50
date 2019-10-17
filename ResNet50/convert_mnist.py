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
