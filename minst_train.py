import tensorflow as tf
import numpy as np
import minst_inference
from tensorflow.examples.tutorials.mnist import input_data
import os

BATCH_SIZE = 100
REGULARAZTION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
MODEL_PATH = "./model"
MODEL_NAME = "model.ckpt"
TRAINING_STEPS = 10000

def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, minst_inference.IMAGE_SIZE,
                                    minst_inference.IMAGE_SIZE, minst_inference.NUM_CHANNELS],
                       name="x-input")
    y_ = tf.placeholder(tf.float32, [None, minst_inference.OUTPUT_NODE], name="y-input")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = minst_inference.inference(x, regularizer, True)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op("train")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshape_xs = np.reshape(xs,(BATCH_SIZE,minst_inference.IMAGE_SIZE,minst_inference.IMAGE_SIZE,minst_inference.NUM_CHANNELS))
            _, loss_val, step = sess.run([train_op, loss, global_step], feed_dict={x: reshape_xs, y_: ys})
            if (i % 1000):
                print("After %d training step(s),loss on training ", "batch is %g." ,step, loss_val)
                saver.save(sess, os.path.join(MODEL_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    minst=input_data.read_data_sets("./data",one_hot="true")
    train(minst)


main()
