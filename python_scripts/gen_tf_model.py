import tensorflow as tf

with tf.Session() as sess:

    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y = tf.placeholder(tf.float32, [None, 10], name="y")

    W = tf.Variable(tf.zeros([784, 10]), name="weights")
    b = tf.Variable(tf.zeros([10]))

    y_out = tf.nn.softmax(tf.matmul(x, W) + b, name="y_out")

    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_out), reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_out))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, name="train")

    correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="test")

    init = tf.variables_initializer(tf.global_variables(), name="init")

    tf.train.write_graph(sess.graph_def,
                         './',
                         'mnist_50_mlp.pb', as_text=False)
