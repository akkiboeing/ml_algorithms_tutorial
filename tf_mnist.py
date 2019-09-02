import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

print(type(mnist))

plt.imshow(mnist.train.images[1].reshape(28,28), cmap = 'gist_gray')
plt.show()

X = tf.placeholder(tf.float32, shape = [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(X,W) + b

y_true = tf.placeholder(tf.float32, shape = [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_true, logits = y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)

train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for step in range(1000):
		batch_x, batch_y = mnist.train.next_batch(100)
		sess.run(train, feed_dict = {X: batch_x, y_true: batch_y})
	matches = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
	accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
	print("Accuracy is: ", sess.run(accuracy, feed_dict = {X: mnist.test.images, y_true: mnist.test.labels}))	

