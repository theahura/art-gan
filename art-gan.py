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

SEED = 123
FLAGS = flags.FLAGS
tfgan = tf.contrib.gan


# DATA.

def _to_patch(im):
    shape = tf.shape(im)
    target = tf.minimum(shape[0], shape[1])
    im = tf.image.resize_image_with_crop_or_pad(im, target, target)
    im = tf.expand_dims(im, axis=0)
    im = tf.image.resize_images(im, [128, 128])
    im = tf.squeeze(im, axis=0)

    im = tf.tile(im, [1, 1, tf.maximum(1, 4 - tf.shape(im)[2])])
    im = tf.slice(im, [0, 0, 0], [128, 128, 3])
    return im


def _open_image(fp):
    im = tf.image.decode_jpeg(tf.read_file(fp), channels=3)
    im = _to_patch(im)
    im = (tf.to_float(im) - 127.5) / 127.5
    return im


def _get_single_dataset(data_dir, batch, seed):
    files = tf.data.Dataset.list_files(data_dir).shuffle(100000000, seed=seed)
    images = files.map(_open_image)
    images = images.apply(tf.contrib.data.batch_and_drop_remainder(batch))
    images = images.prefetch(1).repeat()
    iterator = images.make_one_shot_iterator()
    return iterator.get_next()


def get_dataset(data_dir, batch, seed):
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

    tfgan.eval.add_cyclegan_image_summaries(cyclegan_model)

    return model


# TRAIN.
def main():
    tf.set_random_seed(SEED)

    x, y = get_dataset(FLAGS.data_dir, FLAGS.batch, SEED)

    model = cyclegan_model(x, y)

    # Possibly make this wgan. Has separate params for gen and discrim loss.
    loss = tfgan.cyclegan_loss(
        model,
        tensor_pool_fn=tfgan.features.tensor_pool
    )

    train_ops = tfgan.gan_train_ops(
        model,
        loss,
        generator_optimizer=tf.train.AdamOptimizer(FLAGS.gen_lr, beta1=0.5),
        discriminator_optimizer=tf.train.AdamOptimizer(FLAGS.dis_lr, beta1=0.5),
        summarize_gradients=True,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
    )

    train_steps = tfgan.GANTrainSteps(1, 1)
    tfgan.gan_train(
        train_ops,
        FLAGS.model_dir,
        get_hooks_fn=tfgan.get_sequential_train_hooks(train_steps),
        hooks=[
            tf.train.StopAtStepHook(num_steps=FLAGS.steps)
        ]
    )


if __name__ == '__main__':
    main()
