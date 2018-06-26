---
layout: post
title:  "tensorflow"
date:   2018-06-21 
categories: tensorflow
---

* [1.skill](#1)
* [2.变量、常量](#2)
* [3.变量、常量](#3)
* [4.变量、常量](#4)
* [5.变量、常量](#5)
* [6.变量、常量](#6)
* [7.变量、常量](#7)


#<h2 id="1">1.skills<h2>
```
a[np.random.choice(5,2)] 
```

#<h2 id="2">2.变量、常量<h2>
```
x = tf.Variable([1,2])
a = tf.constant([3,3])

sub = tf.subtract(x,a)

add = tf.add(x,sub)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))
```
#<h2 id="3">3.循环递增<h2>

```
state = tf.Variable(0,name="counter")
new_value = tf.add(state,1)
update = tf.assign(state,new_value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
```
#<h2 id="4">4.fetch、feed<h2>
```
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2,input3)
mul = tf.multiply(input1,add)

with tf.Session() as sess:
	result = sess.run([mul,add])
	print(result)
```

```
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
	print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))
```
#<h2 id="5">5.简单使用示例<h2>
```
import tensorflow as tf
import numpy as np

x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)

init = tf.global_variables_inittializer()

with tf.Session() as sess:
	sess.run(init)
	for step in range(201):
		sess.run(train)
		if step%20 == 0:
			print(step,sess.run([k,b]))
```

#<h2 id="6">6.非线性回归<h2>
```
import tensorflow as tf
import numpy as np

x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zero([1,10]))

Wx_plus_b_L1 = tf.matmul(x,Weights_L1)+biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2

prediction = tf.nn.tanh(Wx_plus_b_L2)

loss = tf.reduce_mean(tf.square(y-prediction))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variable_initializer())
	
	for _ in range(2000):
		sess.run(train_step,feed_dict={x:x_data,y:y_data})

	prediction_value = sess.run(prediction,feed_dict={x:x_data})
	
	plt.figure()
	plt.scatter(x_data,y_data)
	plt.plot(x_data,prediction_value,'r-',lw=5)
	plt.show()

```
#<h2 id="7">7.非线性回归<h2>
```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_vals = np.random.normal(1,0.1,100)
y_vals = np.repeat(10.,100)

x_data = tf.placeholder(shape = [100,],dtype=tf.float32)
y_target = tf.placeholder(shape = [100,],dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape = [100,]))

my_output = tf.multiply(x_data,A)

loss = tf.square(my_output - y_target)

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)


init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(10000): 
        rand_index = np.random.choice(100)

        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]
        sess.run(train_step,feed_dict={x_data:x_vals,y_target:y_vals})


    print(x_vals)
    print(sess.run(A))
    prediction_value = sess.run(my_output,feed_dict={x_data:x_vals})
    print(prediction_value)
    plt.figure()
    plt.scatter(x_vals,y_vals)
    plt.plot(x_vals,prediction_value,'r-',lw=5)
    plt.show()

```

#<h2 id="8">8.线性回归<h2>

```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_vals = np.random.normal(1,0.1,100)
y_vals = np.repeat(10.,100)

x_data = tf.placeholder(shape = [100,],dtype=tf.float32)
y_target = tf.placeholder(shape = [100,],dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape = [100,]))

my_output = tf.multiply(x_data,A)

loss = tf.square(my_output - y_target)

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)


init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(10000): 
        rand_index = np.random.choice(100)

        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]
        sess.run(train_step,feed_dict={x_data:x_vals,y_target:y_vals})


    print(x_vals)
    print(sess.run(A))
    prediction_value = sess.run(my_output,feed_dict={x_data:x_vals})
    print(prediction_value)
    plt.figure()
    plt.scatter(x_vals,y_vals)
    plt.plot(x_vals,prediction_value,'r-',lw=5)
    plt.show()
```
#<h2 id="9">9.iris线性回归<h2>
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf

iris=datasets.load_iris()
# print(iris.target)
binary_target = np.array([1. if x == 0 else 0. for x in iris.target])
iris_2d = np.array([[x[2],x[3]] for x in iris.data])
print(iris_2d)

batch_size = 20
x1_data = tf.placeholder(shape=[None,1],dtype = tf.float32)
x2_data = tf.placeholder(shape=[None,1],dtype = tf.float32)
y_target = tf.placeholder(shape=[None,1],dtype = tf.float32)

A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

my_mult = tf.matmul(x2_data,A)
my_add = tf.add(my_mult,b)
my_output = tf.subtract(x1_data,my_add)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output,labels=y_target)
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(1000):
        rand_index = np.random.choice(len(iris_2d),size=batch_size)
        rand_x = iris_2d[rand_index]
        
        rand_x1 = np.array([[x[0]] for x in rand_x])
        rand_x2 = np.array([[x[1]] for x in rand_x])
        
        print(rand_x1,rand_x2)
        
        rand_y = np.array([[y] for y in binary_target[rand_index]])
        
        sess.run(train_step,feed_dict={x1_data:rand_x1,x2_data:rand_x2,y_target:rand_y})
        
        if (i+1)%200 == 0:
            print(str(sess.run(A)),str(sess.run(b)))
    
    [[slope]] = sess.run(A)
    [[intercept]] = sess.run(b)
    x = np.linspace(0,3,num=50)
    ablineValues = []
    for i in x:
        ablineValues.append(slope*i+intercept)
        
    setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i] == 1]
    setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i] == 1]

    non_setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i] == 0]
    non_setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i] == 0]

    plt.plot(setosa_x,setosa_y,'rx',ms=10,mew=2,label="setosa")
    plt.plot(non_setosa_x,non_setosa_y,'ro',label="Non-setosa")
    plt.plot(x,ablineValues,'b-')
    plt.xlim([0.0,2.7])
    plt.ylim([0.0,7.1])
    plt.xlabel('pl')
    plt.ylabel('pw')
    plt.legend(loc='lower right')
    plt.show()
```

#<h2 id="10">10.conv2d<h2>
```
第二个参数：my_filter [filter_height, filter_width, in_channels,out_channels]
[卷积核高度，卷积核宽度，图像通道数，卷积核个数]
第三个参数：strides=【1,2,2,1】，每个方向上的指定步长，strides在官方定义中是一个一维具有四个元素的张量，其规定前后必须为1，这点大家就别纠结了，所以我们可以改的是中间两个数，中间两个数分别代表了水平滑动和垂直滑动步长值，于是就很好理解了。
```
```
实现一个简单的卷积神经网络：

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_val = np.random.uniform(size=[1,4,4,1])
x_data = tf.placeholder(tf.float32,shape=[1,4,4,1])

my_filter = tf.constant(0.25,shape=[2,2,1,1])
my_strides = [1,2,2,1]
mov_avg_layer = tf.nn.conv2d(x_data,my_filter,my_strides,padding='SAME',name = 'Moving_Avg_Window')

def custom_layer(input_matrix):
    input_matrix_sqeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1.,2.],[-1.,3.]])
    b = tf.constant(1.,shape=[2,2])
    temp1 = tf.matmul(A,input_matrix_sqeezed)
    temp = tf.add(temp1,b)
    return (tf.sigmoid(temp))

with tf.name_scope('Custom_Layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)

with tf.Session() as sess:
    print(sess.run(custom_layer1,feed_dict={x_data:x_val}))
```