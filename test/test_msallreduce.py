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


    def test_horovod_allreduce_cpu(self):
        """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors."""
        tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        tf.contrib.util.constant_value(tensor)
        hvd.init()
        size = hvd.size()
        dtype = tf.float32
        dim = 2
        summed = hvd.allreduce(tensor, average=False)
        diff = self.evaluate(summed)
        print(diff)
if __name__ == '__main__':
    tf.test.main()
