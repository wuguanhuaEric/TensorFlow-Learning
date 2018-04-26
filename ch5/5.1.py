#coding=utf-8
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

#载入MNIST数据集
mnist = input_data.read_data_sets("/path/to/MNIST_data/",one_hot=True)

#打印Training data size: 55000
print "Training data size:", mnist.train.num_examples

#打印Validating data size: 5000
print "Validating data size:", mnist.validation.num_examples

#打印Testing data size: 10000
print "Testing data size:", mnist.test.num_examples

#打印Example training data: [0. 0. 0. ...]
print "Example training data:",mnist.train.images[0]

#打印Example training data label:
#[ 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
print "Example training data label:", mnist.train.labels[0]
