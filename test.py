import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



def make_model(input_dim=784, hidden_dim=64):
    with tf.variable_scope('neural_network'):
        x = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name='input')
        layer_1 = tf.layers.dense(x, hidden_dim, activation=tf.nn.tanh, name='hidden')
        layer_2 = tf.layers.dense(layer_1, 10, name='output')
        prediction = tf.argmax(layer_2, axis=-1, name='prediction')
        return x, prediction, layer_2


def make_conv_model():
    with tf.variable_scope('conv_network'):
        x = tf.placeholder(dtype=tf.float32, shape=(None, 784), name='input')
        x_ = tf.reshape(x, (-1, 28, 28, 1))
        conv_1 = tf.layers.conv2d(x_, 32, kernel_size=(5, 5), padding='same',
                                        activation=tf.nn.relu)
        pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=2)
        conv_2 = tf.layers.conv2d(pool_1, 64, kernel_size=(5, 5), activation=tf.nn.relu)
        pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(2, 2), strides=2)

        flatten = tf.layers.flatten(pool_2)
        dense_1 = tf.layers.dense(flatten, 1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(dense_1, rate=0.4)
        output = tf.layers.dense(dropout, 10, name='output')
        prediction = tf.argmax(output, axis=-1, name='prediction')
        return x, prediction, output

if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('data', one_hot=True)

    n_epochs = 20
    batch_size = 64
    n_examples = mnist.train.images.shape[0]
    n_iter = n_examples // batch_size

    x, prediction, output = make_conv_model()
    y = tf.placeholder(dtype=tf.float32)
    cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)

    optimiser = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(int(n_epochs)):
            losses = []
            for j in range(int(n_iter)):
                images, labels = mnist.train.next_batch(batch_size)
                _, loss = sess.run([optimiser, cost], feed_dict={x:images, y:labels})
                losses.append(loss)
            print("epoch: {}, cost: {}".format(i, np.mean(losses)))

        test_x, test_y = mnist.test.images, mnist.test.labels
        test_pred = sess.run(prediction, feed_dict={x:test_x})
        print(np.mean(test_pred == np.argmax(test_y, axis=-1)))
