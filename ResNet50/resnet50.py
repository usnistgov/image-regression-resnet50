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

class ResNet50():
    _L2_WEIGHT_DECAY = 1e-4

    @staticmethod
    def _gen_l2_regularizer(use_l2_regularizer=True):
        return tf.keras.regularizers.l2(ResNet50._L2_WEIGHT_DECAY) if use_l2_regularizer else None

    @staticmethod
    def _identity_block(input, filters, use_l2_regularizer):
        filter1, filter2, filter3 = filters

        x = tf.keras.layers.Conv2D(
            filters=filter1,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=ResNet50._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(input)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(
            filters=filter2,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=ResNet50._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(
            filters=filter3,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=ResNet50._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.add([x, input])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    @staticmethod
    def _conv_block(input, filters, stride, use_l2_regularizer):
        filter1, filter2, filter3 = filters

        x = tf.keras.layers.Conv2D(
            filters=filter1,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=ResNet50._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(input)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(
            filters=filter2,
            kernel_size=3,
            strides=stride,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=ResNet50._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(
            filters=filter3,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=ResNet50._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        shortcut = tf.keras.layers.Conv2D(
            filters=filter3,
            kernel_size=1,
            strides=stride,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=ResNet50._gen_l2_regularizer(use_l2_regularizer),
            data_format='channels_first')(input)
        shortcut = tf.keras.layers.BatchNormalization(axis=1)(shortcut)

        x = tf.keras.layers.add([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def __init__(self, global_batch_size, img_size, learning_rate=3e-4, use_l2_regularizer=True):

        self.img_size = img_size
        self.learning_rate = learning_rate
        self.global_batch_size = global_batch_size
        self.use_l2_regularizer = use_l2_regularizer

        # image is HWC (normally e.g. RGB image) however data needs to be NCHW for network
        self.inputs = tf.keras.Input(shape=(img_size[2], None, None))
        # self.inputs = tf.keras.Input(shape=(img_size[2], img_size[0], img_size[1]))
        self.model = self._build_model()

        self.loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _build_model(self):
        # reinterpreted from: https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet_model.py

        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=7,
            strides=2,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=ResNet50._gen_l2_regularizer(self.use_l2_regularizer),
            data_format='channels_first')(self.inputs)
        x = tf.keras.layers.BatchNormalization(
            axis=1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        x = ResNet50._conv_block(x, [64, 64, 256], stride=1, use_l2_regularizer=self.use_l2_regularizer)
        x = ResNet50._identity_block(x, [64, 64, 256],
                                 use_l2_regularizer=self.use_l2_regularizer)
        x = ResNet50._identity_block(x, [64, 64, 256],
                                 use_l2_regularizer=self.use_l2_regularizer)

        x = ResNet50._conv_block(x, [128, 128, 512], stride=2,
                                 use_l2_regularizer=self.use_l2_regularizer)
        x = ResNet50._identity_block(x, [128, 128, 512],
                                     use_l2_regularizer=self.use_l2_regularizer)
        x = ResNet50._identity_block(x, [128, 128, 512],
                                     use_l2_regularizer=self.use_l2_regularizer)
        x = ResNet50._identity_block(x, [128, 128, 512],
                                     use_l2_regularizer=self.use_l2_regularizer)

        x = ResNet50._conv_block(x, [256, 256, 1024], stride=2,
                                 use_l2_regularizer=self.use_l2_regularizer)
        x = ResNet50._identity_block(x, [256, 256, 1024],
                                     use_l2_regularizer=self.use_l2_regularizer)
        x = ResNet50._identity_block(x, [256, 256, 1024],
                                     use_l2_regularizer=self.use_l2_regularizer)
        x = ResNet50._identity_block(x, [256, 256, 1024],
                                     use_l2_regularizer=self.use_l2_regularizer)
        x = ResNet50._identity_block(x, [256, 256, 1024],
                                     use_l2_regularizer=self.use_l2_regularizer)
        x = ResNet50._identity_block(x, [256, 256, 1024],
                                     use_l2_regularizer=self.use_l2_regularizer)

        x = ResNet50._conv_block(x, [512, 512, 2048], stride=2,
                                 use_l2_regularizer=self.use_l2_regularizer)
        x = ResNet50._identity_block(x, [512, 512, 2048],
                                     use_l2_regularizer=self.use_l2_regularizer)
        x = ResNet50._identity_block(x, [512, 512, 2048],
                                     use_l2_regularizer=self.use_l2_regularizer)

        # output_layer_name5 is tensor with shape <batch_size>, 2048, <img_size>/32, <img_size>/32
        # downsample_factor = 32
        rm_axes = [2, 3]
        x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, rm_axes), name='reduce_mean')(x)

        logits = tf.keras.layers.Dense(
            1,
            kernel_initializer='he_normal',
            kernel_regularizer=ResNet50._gen_l2_regularizer(self.use_l2_regularizer),
            bias_regularizer=ResNet50._gen_l2_regularizer(self.use_l2_regularizer),
            activation=None,
            name='logits')(x)

        resnet50 = tf.keras.Model(self.inputs, logits, name='resnet50')

        return resnet50

    def get_keras_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def set_learning_rate(self, learning_rate):
        self.optimizer.learning_rate = learning_rate

    def get_learning_rate(self):
        return self.optimizer.learning_rate

    def train_step(self, inputs):
        (images, labels, loss_metric) = inputs
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)

            loss_value = self.loss_fn(labels, logits) # [Nx1]
            # average across the batch (N) with the appropriate global batch size
            loss_value = tf.reduce_sum(loss_value, axis=0) / self.global_batch_size

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, self.model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        loss_metric.update_state(loss_value)

        return loss_value

    @tf.function
    def dist_train_step(self, dist_strategy, inputs):
        per_gpu_loss = dist_strategy.experimental_run_v2(self.train_step, args=(inputs,))
        loss_value = dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_gpu_loss, axis=None)

        return loss_value

    def test_step(self, inputs):
        (images, labels, loss_metric) = inputs
        logits = self.model(images, training=False)

        loss_value = self.loss_fn(labels, logits)
        # average across the batch (N) with the approprite global batch size
        loss_value = tf.reduce_sum(loss_value, axis=0) / self.global_batch_size

        loss_metric.update_state(loss_value)

        return loss_value

    @tf.function
    def dist_test_step(self, dist_strategy, inputs):
        per_gpu_loss = dist_strategy.experimental_run_v2(self.test_step, args=(inputs,))
        loss_value = dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_gpu_loss, axis=None)
        return loss_value
