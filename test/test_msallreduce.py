from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework import ops
import warnings

import horovod.tensorflow as hvd

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class MPITests(tf.test.TestCase):
    """
    Tests for ops in horovod.tensorflow.
    """
    def __init__(self, *args, **kwargs):
        super(MPITests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')
    def evaluate(self, tensors):
        sess = ops.get_default_session()
        if sess is None:
            with self.test_session(config=config) as sess:
                return sess.run(tensors)
        else:
            return sess.run(tensors)


    def test_horovod_multiple_allreduce_cpu(self):
        """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        with tf.device("/cpu:0"):
            if hvd.rank() == 0:
                tensors = [tf.constant([[1.0, 2.0], [3.0, 4.0]]),tf.constant([[9.0, 10.0], [11.0, 12.0]])]
            else:
                tensors = [tf.constant([[5.0, 6.0], [7.0, 8.0]]), tf.constant([[13.0, 14.0], [15.0, 16.0]])]
            summed = 0
            for tensor in tensors:
                summed += hvd.allreduce(tensor, average=False)
        diff = self.evaluate(summed)
        print(diff)
    def test_horovod_single_allreduce_cpu(self):
        """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        with tf.device("/cpu:0"):
            if hvd.rank() == 0:
                tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            else:
                tensor = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            summed = hvd.allreduce(tensor, average=False)
        diff = self.evaluate(summed)
        print(diff)

    def test_horovod_multithread_init(self):
        """Test thread pool init"""
        hvd.init()

if __name__ == '__main__':
    tf.test.main()
