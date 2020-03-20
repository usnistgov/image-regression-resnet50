# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 Required')

import os
# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi

import argparse
import datetime
import numpy as np

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    raise RuntimeError('Tensorflow 2.x.x required')
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import model
import imagereader
import time


def train_model(output_folder, batch_size, reader_count, train_lmdb_filepath, test_lmdb_filepath, use_augmentation, learning_rate, test_every_n_steps, early_stopping_count):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    training_checkpoint_filepath = None

    # setup mixed precision training to use FP16
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    # uses all available devices
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():

        # scale the batch size based on the GPU count
        global_batch_size = batch_size * mirrored_strategy.num_replicas_in_sync
        # scale the number of I/O readers based on the GPU count
        reader_count = reader_count * mirrored_strategy.num_replicas_in_sync

        print('Setting up test image reader')
        test_reader = imagereader.ImageReader(test_lmdb_filepath, use_augmentation=False, shuffle=False, num_workers=reader_count)
        print('Test Reader has {} images'.format(test_reader.get_image_count()))

        print('Setting up training image reader')
        train_reader = imagereader.ImageReader(train_lmdb_filepath, use_augmentation=use_augmentation, shuffle=True, num_workers=reader_count)
        print('Train Reader has {} images'.format(train_reader.get_image_count()))

        try:  # if any errors happen we want to catch them and shut down the multiprocess readers
            print('Starting Readers')
            train_reader.startup()
            test_reader.startup()

            train_dataset = train_reader.get_tf_dataset()
            train_dataset = train_dataset.batch(global_batch_size).prefetch(reader_count)
            train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)

            test_dataset = test_reader.get_tf_dataset()
            test_dataset = test_dataset.batch(global_batch_size).prefetch(reader_count)
            test_dataset = mirrored_strategy.experimental_distribute_dataset(test_dataset)

            print('Creating model')
            renset = model.ResNet50(global_batch_size, train_reader.get_image_size(), learning_rate)

            checkpoint = tf.train.Checkpoint(optimizer=renset.get_optimizer(), model=renset.get_keras_model())

            # train_epoch_size = train_reader.get_image_count()/batch_size
            train_epoch_size = test_every_n_steps
            test_epoch_size = test_reader.get_image_count() / batch_size

            test_loss = list()

            # Prepare the metrics.
            train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
            test_loss_metric = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

            current_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            train_log_dir = os.path.join(output_folder, 'tensorboard-' + current_time, 'train')
            if not os.path.exists(train_log_dir):
                os.makedirs(train_log_dir)
            test_log_dir = os.path.join(output_folder, 'tensorboard-' + current_time, 'test')
            if not os.path.exists(test_log_dir):
                os.makedirs(test_log_dir)

            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            test_summary_writer = tf.summary.create_file_writer(test_log_dir)

            epoch = 0
            print('Running Network')
            while True:  # loop until early stopping
                print('---- Epoch: {} ----'.format(epoch))

                if epoch == 0:
                    cur_train_epoch_size = min(1000, train_epoch_size)
                    print('Performing Adam Optimizer learning rate warmup for {} steps'.format(cur_train_epoch_size))
                    renset.set_learning_rate(learning_rate / 10)
                else:
                    cur_train_epoch_size = train_epoch_size
                    renset.set_learning_rate(learning_rate)

                # Iterate over the batches of the train dataset.
                start_time = time.time()
                for step, (batch_images, batch_labels) in enumerate(train_dataset):
                    if step > cur_train_epoch_size:
                        break

                    inputs = (batch_images, batch_labels, train_loss_metric)
                    renset.dist_train_step(mirrored_strategy, inputs)

                    print('Train Epoch {}: Batch {}/{}: Loss {}'.format(epoch, step, train_epoch_size, train_loss_metric.result()))
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss_metric.result(), step=int(epoch * train_epoch_size + step))
                    train_loss_metric.reset_states()

                # Iterate over the batches of the test dataset.
                epoch_test_loss = list()
                for step, (batch_images, batch_labels) in enumerate(test_dataset):
                    if step > test_epoch_size:
                        break

                    inputs = (batch_images, batch_labels, test_loss_metric)
                    loss_value = renset.dist_test_step(mirrored_strategy, inputs)

                    epoch_test_loss.append(loss_value.numpy())
                    # print('Test Epoch {}: Batch {}/{}: Loss {}'.format(epoch, step, test_epoch_size, loss_value))
                test_loss.append(np.mean(epoch_test_loss))

                print('Test Epoch: {}: Loss = {}'.format(epoch, test_loss_metric.result()))
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', test_loss_metric.result(), step=int((epoch+1) * train_epoch_size))
                test_loss_metric.reset_states()

                with open(os.path.join(output_folder, 'test_loss.csv'), 'w') as csvfile:
                    for i in range(len(test_loss)):
                        csvfile.write(str(test_loss[i]))
                        csvfile.write('\n')

                print('Epoch took: {} s'.format(time.time() - start_time))

                # determine if to record a new checkpoint based on best test loss
                if (len(test_loss) - 1) == np.argmin(test_loss):
                    # save tf checkpoint
                    print('Test loss improved: {}, saving checkpoint'.format(np.min(test_loss)))
                    # checkpoint.save(os.path.join(output_folder, 'checkpoint', "ckpt")) # does not overwrite
                    training_checkpoint_filepath = checkpoint.write(os.path.join(output_folder, 'checkpoint', "ckpt"))

                # determine early stopping
                CONVERGENCE_TOLERANCE = 1e-4
                print('Best Current Epoch Selection:')
                print('Test Loss:')
                print(test_loss)
                min_test_loss = np.min(test_loss)
                error_from_best = np.abs(test_loss - min_test_loss)
                error_from_best[error_from_best < CONVERGENCE_TOLERANCE] = 0
                best_epoch = np.where(error_from_best == 0)[0][0] # unpack numpy array, select first time since that value has happened
                print('Best epoch: {}'.format(best_epoch))

                if len(test_loss) - best_epoch > early_stopping_count:
                    break  # break the epoch loop
                epoch = epoch + 1

        finally: # if any erros happened during training, shut down the disk readers
            print('Shutting down train_reader')
            train_reader.shutdown()
            print('Shutting down test_reader')
            test_reader.shutdown()

    # convert training checkpoint to the saved model format
    if training_checkpoint_filepath is not None:
        # restore the checkpoint and generate a saved model
        renset = model.ResNet50(global_batch_size, train_reader.get_image_size(), learning_rate)
        checkpoint = tf.train.Checkpoint(optimizer=renset.get_optimizer(), model=renset.get_keras_model())
        checkpoint.restore(training_checkpoint_filepath)
        tf.saved_model.save(renset.get_keras_model(), os.path.join(output_folder, 'saved_model'))


def main():
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='train_resnet50', description='Script which trains a ResNet50 Regression model')

    parser.add_argument('--train_database', dest='train_database_filepath', type=str, help='lmdb database to use for (Required)', required=True)
    parser.add_argument('--test_database', dest='test_database_filepath', type=str, help='lmdb database to use for testing (Required)', required=True)
    parser.add_argument('--output_dir', dest='output_folder', type=str, help='Folder where outputs will be saved (Required)', required=True)

    parser.add_argument('--batch_size', dest='batch_size', type=int, help='training batch size', default=4)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=3e-4)
    parser.add_argument('--test_every_n_steps', dest='test_every_n_steps', type=int, help='number of gradient update steps to take between test epochs', default=1000)
    parser.add_argument('--use_augmentation', dest='use_augmentation', type=int, help='whether to use data augmentation [0 = false, 1 = true]', default=1)

    parser.add_argument('--early_stopping', dest='early_stopping_count', type=int, help='Perform early stopping when the test loss does not improve for N epochs.', default=10)
    parser.add_argument('--reader_count', dest='reader_count', type=int, help='how many threads to use for disk I/O and augmentation per gpu', default=1)

    args = parser.parse_args()
    batch_size = args.batch_size
    output_folder = args.output_folder
    early_stopping_count = args.early_stopping_count
    train_lmdb_filepath = args.train_database_filepath
    test_lmdb_filepath = args.test_database_filepath
    learning_rate = args.learning_rate
    test_every_n_steps = args.test_every_n_steps
    use_augmentation = args.use_augmentation
    reader_count = args.reader_count

    print('Arguments:')
    print('batch_size = {}'.format(batch_size))
    print('learning_rate = {}'.format(learning_rate))
    print('test_every_n_steps = {}'.format(test_every_n_steps))
    print('use_augmentation = {}'.format(use_augmentation))

    print('train_database = {}'.format(train_lmdb_filepath))
    print('test_database = {}'.format(test_lmdb_filepath))
    print('output folder = {}'.format(output_folder))

    print('early_stopping count = {}'.format(early_stopping_count))
    print('reader_count = {}'.format(reader_count))

    train_model(output_folder, batch_size, reader_count, train_lmdb_filepath, test_lmdb_filepath, use_augmentation, learning_rate, test_every_n_steps, early_stopping_count)


if __name__ == "__main__":
    main()
