import tensorflow as tf
import sys
import time
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
This example demonstrates asynchronous training
See more: https://www.tensorflow.org/deploy/distributed
'''

FLAGS = tf.app.FLAGS
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

parameter_servers = ["192.168.0.124:2222"]
workers = ["192.168.0.112:2222"]

cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)

batch_size = 100
learning_rate = 0.001
training_epochs = 20
logs_path = "/tmp/mnist/1"

if FLAGS.job_name == 'ps':
    server.join()
elif FLAGS.job_name == 'worker':
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d/cpu:0" % FLAGS.task_index,
        cluster=cluster
    )):
        # Build model...
        global_step = tf.contrib.framework.get_or_create_global_step()

        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
            y = tf.placeholder(tf.float32, Shape=[None, 10], nmae="y")

        tf.set_random_seed(1)
        with tf.name_scope('weights'):
            W1 = tf.Variable(tf.random_normal([784, 100]))
            W2 = tf.Variable(tf.random_normal([100, 10]))

        with tf.name_scope('bias'):
            b1 = tf.Variable(tf.zeros([100]))
            b2 = tf.Variable(tf.zeros([10]))

        with tf.name_scope('softmax'):
            z2 = tf.add(tf.matmul(x,W1), b1)
            a2 = tf.nn.sigmoid(z2)
            z3 = tf.add(tf.matmul(x,W2), b2)
            y_out = tf.nn.softmax(z3)

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_out)

        # build train ops
        with tf.name_scope('train'):
            grad_op = tf.train.MomentumOptimizer(learning_rate, 0.9)

            train_op = grad_op.minimize(cross_entropy, global_step=global_step)

        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        tf.scalar_summary("cost", cross_entropy)
        tf.scalar_summary("accuracy", accuracy)

        summary_op = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()
        print("Variables initialized")

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             global_step=global_step,
                             init_op=init_op)

    begin_time = time.time()
    frequency = 100
    with sv.prepare_or_wait_for_session(server.target) as sess:
        writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())

        start_time = time.time()
        for epoch in range(0, training_epochs):
            batch_count = int(mnist.train.num_examples / batch_size)

            count = 0
            for i in range(0, batch_count):
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                _, cost, summaries, step = sess.run([train_op, cross_entropy,
                                                     summary_op, global_step],
                                                    feed_dict={x: batch_x, y:batch_y})

                writer.add_summary(summaries, step)
                count += 1

                if count % frequency == 0 or (i+1) == batch_count:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print("Step: %d" % (step+1),
                          " Epoch: %2d," % (epoch + 1),
                          " Batch: %3d of %3d," % (i+1, batch_count),
                          " Cost: %.4f," % cost,
                          " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))

                    count = 0

        print("Test Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.lables}))
        print("Total time:%3.2fs") % float(time.time() - begin_time)
        print("Final Cost: %.4f" % cost)

    sv.stop()
    print("done")
