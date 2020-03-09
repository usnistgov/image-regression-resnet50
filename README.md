# Regression

Regression Tesnorflow model ready to run on Enki.
- ResNet50: [https://arxiv.org/pdf/1512.03385.pdf](https://arxiv.org/pdf/1512.03385.pdf)


Enki AI Cluster page: 
- [https://aihpc.ipages.nist.gov/pages/](https://aihpc.ipages.nist.gov/pages/)
- [https://gitlab.nist.gov/gitlab/aihpc/pages/wikis/home](https://gitlab.nist.gov/gitlab/aihpc/pages/wikis/home)

*This codebase is designed to work with Python3 and Tensorflow 2.x*


# Input Data Constraints
There is example input data included in the repo under the [data](https://gitlab.nist.gov/gitlab/mmajursk/Regression/tree/master/data) folder

Input data assumptions:
- image type: grayscale image with one of these pixel types: uint8, uint16, int32, float32
- Regresion value type: single number of type: uint8, uint16, int32, flaot32
- each input image must have a corresponding regression target value

Before training script can be launched, the input data needs to be converted into a memory mapped database ([lmdb](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database)) to enable fast memory mapped file reading during training. 

# LMDB Construction
This training code uses lmdb databases to store the image and mask data to enable parallel memory-mapped file reader to keep the GPUs fed. 

The input folder of images and masks needs to be split into train and test. Train to update the model parameters, and test to estimate the generalization accuracy of the resulting model. By default 80% of the data is used for training, 20% for test.


```
python build_lmdb.py -h
usage: build_lmdb [-h] [--image_folder IMAGE_FOLDER]
                  [--csv_filepath CSV_FILEPATH]
		  [--output_filepath OUTPUT_FILEPATH]
                  [--dataset_name DATASET_NAME]
                  [--train_fraction TRAIN_FRACTION]

Script which converts two folders of images and masks into a pair of lmdb
databases for training.

optional arguments:
  -h, --help            show this help message and exit
  --image_folder IMAGE_FOLDER
                        filepath to the folder containing the images
  --csv_filepath CSV_FILEPATH
                        filepath to the file containing the ground truth labels. Csv file SHOULD NOT HAVE HEADERS!
  --output_filepath OUTPUT_FILEPATH
                        filepath to the folder where the outputs will be
                        placed
  --dataset_name DATASET_NAME
                        name of the dataset to be used in creating the lmdb
                        files
  --train_fraction TRAIN_FRACTION
                        what fraction of the dataset to use for training
```


# Training
With the lmdb build there are two methods for training a model. 

1. Cluster: Single Node Multi GPU
	- the include shell script `launch_train_sbatch.sh` is setup to all training on the Enki Cluster.
2. Local: Single Node Multi GPU
	- If you want to train the model on local hardware, avoid using `launch_train_sbatch.sh`, use python and directly launch `train_resnet50.py`.

Whether you launch the training locally or on Enki, the training script will query the system and determine how many GPUs are available. It will then build the network for training using data-parallelism. So when you define your batch size, it will actually be multiplied by the number of GPUs you are using to train the network. So a batch size of 8 training on 4 gpus, results in an actual batch size of 32. Each GPU computes its own forward pass on its own data, computes the gradient, and then all N gradients are averaged. The gradient averaging can either happen on the CPU, or on any of the GPUs. 

On Enki, the GPUs have a fully connected topology:

```
2019-02-27 13:05:03.068185: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0, 1, 2, 3
2019-02-27 13:05:04.525934: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-02-27 13:05:04.527345: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 1 2 3 
2019-02-27 13:05:04.528317: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N Y Y Y 
2019-02-27 13:05:04.529605: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 1:   Y N Y Y 
2019-02-27 13:05:04.530902: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 2:   Y Y N Y 
2019-02-27 13:05:04.532257: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 3:   Y Y Y N 
```

However, on other systems this might not be the case:

```
2019-02-27 13:45:03.442171: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1471] Adding visible gpu devices: 0, 1, 2, 3
2019-02-27 13:45:05.405248: I tensorflow/core/common_runtime/gpu/gpu_device.cc:952] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-02-27 13:45:05.405293: I tensorflow/core/common_runtime/gpu/gpu_device.cc:958]      0 1 2 3 
2019-02-27 13:45:05.405302: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   N Y N N 
2019-02-27 13:45:05.405307: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 1:   Y N N N 
2019-02-27 13:45:05.405311: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 2:   N N N Y 
2019-02-27 13:45:05.405315: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 3:   N N Y N 
```

The full help for the training script is:


```
python train.py -h
usage: train [-h] --train_database TRAIN_DATABASE_FILEPATH
                      --test_database TEST_DATABASE_FILEPATH --output_dir
                      OUTPUT_FOLDER [--batch_size BATCH_SIZE]
                      [--learning_rate LEARNING_RATE]
                      [--test_every_n_steps TEST_EVERY_N_STEPS]
                      [--use_augmentation USE_AUGMENTATION]
                      [--early_stopping EARLY_STOPPING_COUNT]
                      [--reader_count READER_COUNT]

Script which trains a ResNet50 Regression model

optional arguments:
  -h, --help            show this help message and exit
  --train_database TRAIN_DATABASE_FILEPATH
                        lmdb database to use for (Required)
  --test_database TEST_DATABASE_FILEPATH
                        lmdb database to use for testing (Required)
  --output_dir OUTPUT_FOLDER
                        Folder where outputs will be saved (Required)
  --batch_size BATCH_SIZE
                        training batch size
  --learning_rate LEARNING_RATE
  --test_every_n_steps TEST_EVERY_N_STEPS
                        number of gradient update steps to take between test
                        epochs
  --use_augmentation USE_AUGMENTATION
                        whether to use data augmentation [0 = false, 1 = true]
  --early_stopping EARLY_STOPPING_COUNT
                        Perform early stopping when the test loss does not
                        improve for N epochs.
  --reader_count READER_COUNT
                        how many threads to use for disk I/O and augmentation
                        per gpu

```

A few of the arguments require explanation.

- `test_every_n_steps`: typically, you run test/validation every epoch. However, I am often building models with very small amounts of data (e.g. 500 images). With an actual batch size of 32, that allows me 15 gradient updates per epoch. The model does not change that fast, so I impose a fixed global step count between test so that I don't spend all of my GPU time running the test data. A good value for this is typically 1000.
- `early_stopping`: this is an integer specifying the early stopping criteria. If the model test loss does not improve after this number of epochs (epoch defined as `test_every_n_steps steps` updates) training is terminated because we have moved into overfitting the training dataset.


# Image Readers
One of the defining features of this codebase is the parallel (python multiprocess) image reading from lightning memory mapped databases. 

There are typically 1 or more reader threads feeding each GPU. 

Each ImageReader class instance:
- selects the next image (potentially at random from the shuffled dataset)
- loads images from a shared lmdb read-only reference
- determines the image augmentation parameters from by defining augmentation limits
- applies the augmentation transformation to the image
- add the augmented image to the batch that reader is building
- once a batch is constructed, the imagereader adds it to the output queue shared among all of the imagereaders

The training script setups of python generators which just get a reference to the output batch queue data and pass it into tensorflow. One of the largest bottlenecks in deep learning is keeping the GPUs fed. By performing the image reading and data augmentation asynchronously all the main python training thread has to do is get a reference to the next batch (which is waiting in memory) and pass it to tensorflow to be copied to the GPUs.

If the imagereaders do not have enough bandwidth to keep up with the GPUs you can increase the number of readers per gpu, though 1 or 2 readers per gpus is often enough. 

You will know whether the image readers are keeping up with the GPUs. When the imagereader output queue is getting empty a warning is printed to the log:

```
Input Queue Starvation !!!!
```

along with the matching message letting you know when the imagereaders have caught back up:

```
Input Queue Starvation Over
```

# Image Augmentation

For each image being read from the lmdb, a unique set of augmentation parameters are defined. 

the `augment` class supports:

| Transformation  | Parameterization |
| ------------- | ------------- |
| reflection (x, y) | Bernoulli  |
| rotation  | Uniform |
| jitter (x, y)  | Percent of Image Size  |
| scale (x,y)  | Percent Change |
| noise  | Percent Change of Current Image Dynamic Range  |
| blur  | Uniform Selection of Kernel Size |
| pixel intensity  | Percent Change of Current Image Dynamic Range |


These augmentation transformations are generally configured based on domain expertise and stay fixed per dataset.

Currently the only method for modifying them is to open the `imagereader.py` file and edit the augmentation parameters contained within the code block within the imagereader `__init__`:

```
# setup the image data augmentation parameters
self._reflection_flag = True
self._rotation_flag = True
self._jitter_augmentation_severity = 0.1  # x% of a FOV
self._noise_augmentation_severity = 0.02  # vary noise by x% of the dynamic range present in the image
self._scale_augmentation_severity = 0.1  # vary size by x%
self._blur_max_sigma = 2  # pixels
# self._intensity_augmentation_severity = 0.05
``` 
