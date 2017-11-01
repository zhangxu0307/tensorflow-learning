
import tensorflow as tf
import numpy as np

s = tf.InteractiveSession()

index = tf.constant([0,1,1,2,2])
x1 = tf.constant([[1,2,3,4],[4,5,6,4],[7,8,9,4],[2,3,4,4],[1,8,0,4]])
x2 = tf.constant([1,2,3,4])
x3 = tf.constant([1,2,3,7,8,9])
boolx = tf.constant([[True,True],[False,True]])
a = tf.zeros([5,30])
b = tf.to_double(a)
print (b)
# print (tf.segment_prod(x1,index).eval())
# print (tf.argmin(x1,axis=1).eval())
# print (tf.where(boolx).eval())
# print (tf.unique(x2)[0].eval())

# x22 = tf.expand_dims(x2,1)
# print (tf.rank(x22).eval())
# print (tf.squeeze(x22).eval())

print (tf.slice(x1,[1,1],[2,2]).eval())
print (tf.pad(x1,[[1,1],[2,2]]).eval())
#print (t1.eval())
