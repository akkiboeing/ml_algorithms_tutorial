import tensorflow as tf
import numpy as np

print(tf.__version__)

hello = tf.constant('Hello World')
x = tf.constant(100)

print(type(x))
print(type(hello))

sess = tf.Session()

sess.run(hello)
print(type(sess.run(hello)))

sess.run(x)
print(type(sess.run(x)))

x = tf.constant(2)
y = tf.constant(3)

# arithmetic operations with constants
with tf.Session() as sess:
    print('Addition',sess.run(x+y))
    print('Subtraction',sess.run(x-y))
    print('Multiplication',sess.run(x*y))
    print('Division',sess.run(x/y))

x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)    

print(x)
print(type(x))

add = tf.add(x,y)
sub = tf.subtract(x,y)
mul = tf.multiply(x,y)

d = {x:20,y:30}

# arithmetic operatins with placeholders
with tf.Session() as sess:
    print('Addition',sess.run(add,feed_dict=d))
    print('Subtraction',sess.run(sub,feed_dict=d))
    print('Multiplication',sess.run(mul,feed_dict=d))

# matric multiplication
a = np.array([[5.0,5.0]])
b = np.array([[2.0],[2.0]])

mat1 = tf.constant(a)
mat2 = tf.constant(b)

matrix_multi = tf.matmul(mat1,mat2)

with tf.Session() as sess:
    result = sess.run(matrix_multi)
    print(result)