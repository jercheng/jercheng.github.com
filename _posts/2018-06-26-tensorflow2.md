---
layout: post
title:  "tensorflow2"
date:   2018-06-26 
categories: tensorflow
---

* [1.skill](#1)
* [2.实现门函数](#2)
* [3.单隐藏层神经网络](#3)



<h2 id="1">1.skills</h2>

```
a[np.random.choice(5,2)] 
```
 * tf.set_random_seed(seed) 图级、操作级；随机数一样

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

<h2 id="3">3.单隐藏层神经网络</h2>

> 全连接神经网络算法主要是基于矩阵乘法

```
实现单层神经网络，5个神经元，3个特征

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

iris_data = datasets.load_iris()
x_vals = np.array([x[0:3] for x in iris_data.data])
y_vals = np.array([x[3] for x in iris_data.data])

seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

train_indices = np.random.choice(len(x_vals),round(len(x_vals)*0.8),replace=False)
test_indices = np.array(list(set(range(len(x_vals)))-set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]

y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

def normalize_cols(m):
    col_max = m.max(axis = 0)
    col_min = m.min(axis = 0)
    
    return (m-col_min)/(col_max-col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

batch_size = 50
x_data = tf.placeholder(shape=[None,3],dtype=tf.float32)
y_target = tf.placeholder(shape=[None,1],dtype=tf.float32)

hidden_layer_nodes = 5
A1 = tf.Variable(tf.random_normal(shape = [3,hidden_layer_nodes],dtype=tf.float32))
b1 = tf.Variable(tf.random_normal(shape = [hidden_layer_nodes]))
A2 = tf.Variable(tf.random_normal(shape = [hidden_layer_nodes,1]))
b2 = tf.Variable(tf.random_normal(shape = [1]))

hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data,A1),b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output,A2),b2))

loss = tf.reduce_mean(tf.square(y_target - final_output))

my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)

init_values = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_values)
    
    loss_vec = []
    test_vec = []
    
    for i in range(1000):
        rand_index = np.random.choice(len(x_vals_train),size=batch_size)
        
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        
        sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
        
        temp_loss = sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y})
        loss_vec.append(np.sqrt(temp_loss))
        
        test_temp_loss = sess.run(loss,feed_dict={x_data:x_vals_test,y_target:np.transpose([y_vals_test])})
        
        test_vec.append(np.sqrt(test_temp_loss))
        
        if (i+1)% 50 == 0:
            print(i+1,temp_loss)
    
    
    plt.plot(loss_vec,'k-',label='train_loss')
    plt.plot(test_vec,'r--',label='test_loss')
    plt.title('a')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
```