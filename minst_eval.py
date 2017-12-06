import tensorflow as tf
import os
import minst_inference
import minst_train
import time
EVAL_INTERVAL_SECS = 10


def evaluate(minst):
    x = tf.placeholder(tf.float32, [None, minst_inference.INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, minst_inference.OUTPUT_NODE], name="y-input")
    validate_feed = ({x: minst.validation.images, y_: minst.validation.labels})
    y = minst_inference.inference(x, None, False)
    correct_prediction = tf.equal(tf.argmax(y), tf.argmax(y_))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    variable_averages = tf.train.ExponentialMovingAverage(minst_train.LEARNING_RATE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saverPath = os.path.join(minst_train.MODEL_PATH, minst_train.MODEL_NAME)
    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(minst_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, saverPath)
                # .split('/')[-1].split('-')[-1]
                global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
            else:
                print ("NO CHECKPOINT FIND")
                return
        time.sleep(EVAL_INTERVAL_SECS)

