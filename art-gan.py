"""
Amol Kapoor

Implementation of Cycle GAN for photo-to-image-style transfer.
"""

import os

from absl import flags
import numpy as np
import tensorflow as tf

# Filepaths.
flags.DEFINE_string('data_dir', 'datasets/monet2photo', 'Data path.')
flags.DEFINE_string('model_dir', 'ckpts', 'model path.')

# Model hyperparameters.
flags.DEFINE_float('gen_lr', 0.0002)
flags.DEFINE_float('dis_lr', 0.0001)

# Training parameters.
flags.DEFINE_integer('steps', 50000)
flags.DEFINE_integer('batch', 64)
flags.DEFINE_inteGger('patch', 128)

SEED = 123
IMSIZE = (256, 256)
FLAGS = flags.FLAGS
tfgan = tf.contrib.gan


# DATA.

def _open_image(fp):
    im = tf.image.decode_jpeg(tf.read_file(fp), channels=3)
    im = tf.image.resize_images(im, IMSIZE)
    im = tf.cast(im, tf.float32)
    return im


def _get_single_dataset(data_dir, batch, seed):
    files = tf.data.Dataset.list_files(data_dir).shuffle(100000000, seed=seed)
    images = files.map(_open_image)
    images = images.apply(tf.contrib.data.batch_and_drop_remainder(batch))
    images = images.prefetch(1).repeat()
    iterator = images.make_one_shot_iterator()
    return iterator.get_next()


def _get_dataset(data_dir, batch, seed):
    fake_fp = os.path.join(data_dir, '/trainA')
    real_fp = os.path.join(data_dir, '/trainB')
    real = _get_single_dataset(real_fp, batch, seed)
    fake = _get_single_dataset(fake_fp, batch, seed)
    return fake, real

# MODEL.
def generator(net):
    pass


def discriminator(net):
    pass


def cyclegan_model(input_set, ground_truth):
    model = tfgan.cyclegan_model(
        generator_fn=generator,
        discriminator_fn=discriminator,
        data_x=input_set,
        data_y=ground_truth
    )

    return model


# TRAIN.
def main():
    tf.set_random_seed(SEED)
    pass


if __name__ == '__main__':
    main()
