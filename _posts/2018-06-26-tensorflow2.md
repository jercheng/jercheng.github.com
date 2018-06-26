---
layout: post
title:  "tensorflow2"
date:   2018-06-26 
categories: tensorflow
---

* [1.skill](#1)
* [2.实现门函数](#2)



<h2 id="1">1.skills</h2>

```
a[np.random.choice(5,2)] 
```

<h2 id="2">2.实现门函数</h2>

```
import tensorflow as tf

x_value = 5.
a = tf.Variable(tf.constant(4.))
x_data = tf.placeholder(dtype=tf.float32)

mul_opt = tf.multiply(a,x_data)
loss = tf.square(tf.subtract(mul_opt,50.))
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

init_value = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_value)
    
    for i in range(10):
        sess.run(train_step,feed_dict={x_data:x_value})
        
        a_val = sess.run(a)
        mul_result = sess.run(mul_opt,feed_dict={x_data:x_value})
        
        print(a_val,mul_result)
```