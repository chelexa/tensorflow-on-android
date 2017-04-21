# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Add a feed weight here. This will be input before training, and after each
  # training step
  w = tf.placeholder(tf.float32, [784, 10])

  # The weight var that will actually be used by the training step. It is simply
  # assigned the samw value as the feed weight.
  W = tf.Variable(tf.identity(w))

  # The below code was taken, unchanged, from github.com/tensorflow/tensorflow

  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  # Need to feed in a value to w (little w, the feed) or else it will error
  tf.global_variables_initializer().run(feed_dict={w: np.zeros((784, 10), dtype=np.float32)})

  # Train
  results = []
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    w_in = None
    if len(results) == 0:
        # init feed
        # NOTE: This may not be necessary, because we already initialized the w
        # feed in line 56 above. TODO(tylermzeller, 3ygun) test to see if this is
        # the case.
        w_in = np.zeros([784, 10])
    else:
        # This is the trained weight after a train step. NOTE: see note above.
        w_in = results[1]
    results = sess.run([train_step, W], feed_dict={x: batch_xs, y_: batch_ys, w: w_in})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
