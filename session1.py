import pandas as pd 
import numpy as np 
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import pdb
import logging.config
import tensorflow as tf 

## Maintain logging.ini. importing logging.config is required to use the config

logging.config.fileConfig("logging.ini")

logger=logging.getLogger(__name__)

def create_model():
    # define the inputs
    with tf.variable_scope('geometric_mean_model'):
        x= tf.placeholder(tf.float32)
        y=tf.placeholder(tf.float32)
        g_mean=tf.sqrt(x*y)
    with tf.variable_scope('linear_sum_model',reuse=tf.AUTO_REUSE):
        w=tf.get_variable("weight",dtype=tf.float32,initializer=tf.constant(0.2))
        model=g_mean+w
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res=sess.run(model, feed_dict={x:2.0,y:8.0})
    return res

def preprocess_data(im, label):
    im=tf.cast(im,tf.float32)
    im=im/127.5
    im=im-1
    im=tf.reshape(im,[-1])
    return im,label


def data_layer(data_tensor,num_threads=4,batch_size=32,prefetch_buffer=100):
    with tf.variable_scope("data"):
        dataset=tf.data.Dataset.from_tensor_slices(data_tensor)
        dataset=dataset.shuffle(buffer_size=60000).repeat() 
        dataset=dataset.map(preprocess_data,num_parallel_calls=num_threads)
        dataset=dataset.batch(batch_size)
        dataset=dataset.prefetch(prefetch_buffer)
        iterator=dataset.make_one_shot_iterator()
    return iterator

def model(input_layer,num_classes=10):
    with tf.variable_scope("model"):
        net=tf.layers.dense(input_layer,512)
        net=tf.nn.relu(net)
        net=tf.layers.dense(net,num_classes) #creating dense layer of 8 units
    return net

def loss_function(logits,labels,num_classes=10):
    with tf.variable_scope('loss'):
        target_prob=tf.one_hot(labels,num_classes)    
        total_loss=tf.losses.softmax_cross_entropy(target_prob,logits)
    return total_loss

def optimizer_func(total_loss,global_step,learning_rate=0.1):
    with tf.variable_scope("optimizer"):
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optmizer=optimizer.minimize(total_loss,global_step)
    return optmizer

def performance_metric(logits,labels):
    with tf.variable_scope("performance_metric"):
        preds=tf.argmax(logits,axis=1) #undoing the one hot encoding
        labels=tf.cast(labels,tf.int64)
        corrects=tf.equal(preds,labels)
        accuracy=tf.reduce_mean(tf.cast(corrects,tf.float32))
    return accuracy

def train(data_tensor):
    global_step=tf.Variable(1, dtype=tf.int32,trainable=False,name="iter_number")
    # Graph
    images,labels=data_layer(data_tensor).get_next()
    logits=model(images)
    loss=loss_function(logits,labels)
    optimizer=optimizer_func(loss,global_step)
    accuracy=performance_metric(logits,labels)
    #training
    num_iter=10000
    log_iter=1000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        streaming_loss=0
        streaming_accuracy=0
        for i in range(1,num_iter):
            _,loss_batch,acc_batch=sess.run([optimizer,loss,accuracy]) #async call
            #loss is redundant here
            streaming_loss+=loss_batch
            streaming_accuracy+=acc_batch
            if i%log_iter==0:
                logging.info("Iteration {}, Loss: {:.2f} Accuracy {:.2f}"
                         .format(i, streaming_loss/log_iter, streaming_accuracy/log_iter))
                streaming_loss=0
                streaming_accuracy=0

if __name__=='__main__':
    logger.info('Exercise 1: Calculate Geometric Mean and Add a constant value')
    result=create_model()
    logger.info(f"Result for Exercise 1 is {result}")
    logger.info('Exercise 2: Classifying Hand Digits with Mnist Dataset')
    data_train,data_val=tf.keras.datasets.mnist.load_data()
    train(data_tensor=data_train)
    logging.info('Result for Exercise 2')
