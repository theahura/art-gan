"""
Amol Kapoor

Implementation of Cycle GAN for photo-to-image-style transfer.

TODO:
  Utilize pretrained imagenet models to avoid training whole thing.
"""

import os

from absl import flags
import numpy as np
import tensorflow as tf
from models.research.slim.nets import cyclegan
from models.research.slim.nets import pix2pix
import IPython

# Filepaths.
flags.DEFINE_string('data_dir', 'datasets/monet2photo', 'Data path.')
flags.DEFINE_string('model_dir', 'ckpts', 'model path.')

# Model hyperparameters.
flags.DEFINE_float('gen_lr', 0.0002, 'Generator learning rate.')
flags.DEFINE_float('dis_lr', 0.0001, 'Discriminator learning rate.')

# Training parameters.
flags.DEFINE_integer('steps', 50000, 'Number of steps to run.')
flags.DEFINE_integer('batch', 1, 'Batch size for images.')

SEED = 123
FLAGS = flags.FLAGS
tfgan = tf.contrib.gan
layers = tf.layers


# DATA.
def _to_patch(im):
  shape = tf.shape(im)
  target = tf.minimum(shape[0], shape[1])
  im = tf.image.resize_image_with_crop_or_pad(im, target, target)
  im = tf.expand_dims(im, axis=0)
  im = tf.image.resize_images(im, [64, 64])
  im = tf.squeeze(im, axis=0)

  im = tf.tile(im, [1, 1, tf.maximum(1, 4 - tf.shape(im)[2])])
  im = tf.slice(im, [0, 0, 0], [64, 64, 3])
  return im


def _open_image(fp):
  im = tf.image.decode_jpeg(tf.read_file(fp), channels=3)
  im = _to_patch(im)
  im = (tf.to_float(im) - 127.5) / 127.5
  return im


def _get_single_dataset(data_dir, batch, seed):
  files = tf.data.Dataset.list_files(data_dir + '/*').shuffle(100000000, seed=seed)
  images = files.map(_open_image)
  images = images.apply(tf.contrib.data.batch_and_drop_remainder(batch))
  images = images.prefetch(3).repeat()
  iterator = images.make_one_shot_iterator()
  return iterator.get_next()


def get_dataset(data_dir, batch, seed):
  fake_fp = os.path.join(data_dir, 'trainA')
  real_fp = os.path.join(data_dir, 'trainB')
  real = _get_single_dataset(real_fp, batch, seed)
  fake = _get_single_dataset(fake_fp, batch, seed)
  return fake, real


# MODEL.
def generator(input_images):
  input_images.shape.assert_has_rank(4)
  input_size = input_images.shape.as_list()
  channels = input_size[-1]
  if channels is None:
    raise ValueError(
      'Last dimension shape must be known but is None: %s' % input_size)
  with tf.contrib.framework.arg_scope(cyclegan.cyclegan_arg_scope()):
    output_images, _ = cyclegan.cyclegan_generator_resnet(input_images,
                                                          num_outputs=channels)
    return output_images


def discriminator(image_batch, condition):
  with tf.contrib.framework.arg_scope(pix2pix.pix2pix_arg_scope()):
    logits_4d, _ = pix2pix.pix2pix_discriminator(
      image_batch, num_filters=[64, 128, 256, 512])
    logits_4d.shape.assert_has_rank(4)
    # Output of logits is 4D. Reshape to 2D, for TFGAN.
    logits_2d = tf.contrib.layers.flatten(logits_4d)
    return logits_2d


def cyclegan_model(input_set, ground_truth):
  model = tfgan.cyclegan_model(
    generator_fn=generator,
    discriminator_fn=discriminator,
    data_x=input_set,
    data_y=ground_truth
  )

  tfgan.eval.add_cyclegan_image_summaries(model)

  return model


def _get_lr(base_lr):
  global_step = tf.train.get_or_create_global_step()
  lr_constant_steps = FLAGS.steps // 2
  def _lr_decay():
    return tf.train.polynomial_decay(
      learning_rate=base_lr,
      global_step=(global_step - lr_constant_steps),
      decay_steps=(FLAGS.steps - lr_constant_steps),
      end_learning_rate=0.0)
  return tf.cond(global_step < lr_constant_steps, lambda: base_lr, _lr_decay)


# TRAIN.
def main(_):
  tf.set_random_seed(SEED)

  x, y = get_dataset(FLAGS.data_dir, FLAGS.batch, SEED)

  model = cyclegan_model(x, y)

  # Possibly make this wgan. Has separate params for gen and discrim loss.
  loss = tfgan.cyclegan_loss(
    model,
    tensor_pool_fn=tfgan.features.tensor_pool
  )

  gen_lr = _get_lr(FLAGS.gen_lr)
  dis_lr = _get_lr(FLAGS.dis_lr)

  train_ops = tfgan.gan_train_ops(
    model,
    loss,
    generator_optimizer=tf.train.AdamOptimizer(gen_lr, beta1=0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(dis_lr, beta1=0.5),
    summarize_gradients=True,
    aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
  )

  train_steps = tfgan.GANTrainSteps(1, 1)
  status_message = tf.string_join(
      [
          'Starting train step: ',
          tf.as_string(tf.train.get_or_create_global_step())
      ], name='status_message')
  tfgan.gan_train(
    train_ops,
    FLAGS.model_dir,
    get_hooks_fn=tfgan.get_sequential_train_hooks(train_steps),
    hooks=[
      tf.train.StopAtStepHook(last_step=FLAGS.steps),
      tf.train.LoggingTensorHook([status_message], every_n_iter=100)
    ],
  )

  IPython.embed()


if __name__ == '__main__':
  tf.app.run(main)
