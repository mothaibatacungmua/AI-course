import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mnist.train
'''
$ python example.py --job-name="ps" --task_index=0
$ python example.py --job-name="worker" --task_index=0


This example demonstrates asynchronous training
See more: https://www.tensorflow.org/deploy/distributed
'''

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

parameter_servers = ["172.30.0.187:2222"]
workers = ["172.30.0.44:2222", "172.30.0.156:2222"]

cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)

batch_size = 100
learning_rate = 0.01
training_epochs = 20
logs_path = "/tmp/mnist/1"

def main(_):
    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == 'worker':
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d/gpu:0" % FLAGS.task_index,
            cluster=cluster
        )):
            # Build model...
            global_step = tf.contrib.framework.get_or_create_global_step()

            x = tf.placeholder(tf.float32, shape=[None, 784])
            y_ = tf.placeholder(tf.float32, shape=[None, 10])
            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))
            y = tf.matmul(x, W) + b
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
            # build train ops
            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)


        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                pass

            def before_run(self, run_context):
                batch_x, batch_y = mnist.train.next_batch(100)
                return tf.train.SessionRunArgs([cross_entropy, global_step],
                                               feed_dict={x:batch_x, y_:batch_y})

            def after_run(self, run_context, run_values):

                lost, global_step = run_values.results

                print("Iter:%d Lost:%0.5f" % (global_step, lost))


        hooks = [tf.train.StopAtStepHook(last_step=10000), _LoggerHook()]
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir="",
                                               hooks=hooks,
                                               config=tf.ConfigProto(allow_soft_placement=True)) \
                as mon_sess:

            while not mon_sess.should_stop():
                mon_sess.run(train_op)

if __name__ == '__main__':
    tf.app.run(main=main)