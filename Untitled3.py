#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np


# In[2]:


def load_data (path_data):
    rows = []
#     with open('Plate_license_train.csv', 'r') as csv_file:
    with open(path_data, 'r') as csv_file:
        result = csv.reader(csv_file)
        # đọc từng dòng của file và thêm vào list rows, mỗi phần tử của list là một dòng
        for row in result:
            rows.append(row)
    data = []
    label = []
    
    for letter in rows :
        x = np.array([int(j) for j in letter[1:]])
        x = x.reshape(28, 28)
        
        data.append(x)
        label.append(int(letter[0]))
            
    return data, label


# In[3]:


path_train = 'Plate_license_train.csv'
path_test = 'Plate_license_test.csv'
path_val = 'Plate_license_val.csv'


data_train = []
label_train = []

data_test = []
label_test = []

data_val = []
label_val = []

data_train, label_train = load_data(path_train)
data_test, label_test = load_data(path_test)
data_val, label_val = load_data(path_val)


# In[30]:


get_ipython().run_line_magic('tensorflow_version', '1.x')
import os
import sys
import shutil
import numpy as np
import tensorflow as tf


# In[5]:


SCRIPT_DIR = os.getcwd()


# In[6]:


TRAIN_GRAPH = 'training_graph.pb'
CHKPT_FILE = 'float_model.ckpt'


# In[7]:


CHKPT_DIR = os.path.join(SCRIPT_DIR, 'chkpts')
TB_LOG_DIR = os.path.join(SCRIPT_DIR, 'tb_logs')
CHKPT_PATH = os.path.join(CHKPT_DIR, CHKPT_FILE)
MNIST_DIR = os.path.join(SCRIPT_DIR, 'mnist_dir')


# In[8]:


if not (os.path.exists(MNIST_DIR)):
    os.makedirs(MNIST_DIR)
    print("Directory " , MNIST_DIR ,  "created ") 


# In[9]:


if (os.path.exists(TB_LOG_DIR)):
    shutil.rmtree(TB_LOG_DIR)
os.makedirs(TB_LOG_DIR)
print("Directory " , TB_LOG_DIR ,  "created ") 


# In[10]:


if (os.path.exists(CHKPT_DIR)):
    shutil.rmtree(CHKPT_DIR)
os.makedirs(CHKPT_DIR)
print("Directory " , CHKPT_DIR ,  "created ")


# In[11]:


LEARN_RATE = 0.0001
BATCHSIZE = 50
EPOCHS = 3


# In[12]:


x_train = data_train
y_train  = label_train

x_test = data_test
y_test = label_test


# In[13]:


print(type(data_train))
x_train = np.reshape(x_train, [-1, 28, 28, 1])
x_test = np.reshape(x_test, [-1, 28, 28, 1])


# In[14]:


x_train = (x_train/255.0)  
x_test = (x_test/255.0)


# In[15]:


# one-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


# In[16]:


x_valid = data_val
y_valid = label_val

x_valid = np.reshape(x_valid, [-1, 28, 28, 1])
x_valid = (x_valid/255.0)
y_valid  = tf.keras.utils.to_categorical(y_valid)


# In[17]:


total_batches = int(len(x_train)/BATCHSIZE)


# In[18]:


x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='images_in')
y = tf.placeholder(tf.float32, [None, 10], name='labels_in')


# In[19]:


def cnn(x):
  '''
  Build the convolution neural network
  arguments:
    inputs: the input tensor - shape must be [None,28,28,1]
  '''
  net = tf.layers.conv2d(x, 16, [3, 3], activation=tf.nn.relu)
  net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
  net = tf.layers.conv2d(net, 32, [3, 3], activation=tf.nn.relu)
  net = tf.layers.flatten(net)
  net = tf.layers.dense(net, units=256, activation=tf.nn.relu)
  logits = tf.layers.dense(net, units=10, activation=None)
  return logits


# In[20]:


# build the network, input comes from the 'x' placeholder
logits = cnn(x)


# In[21]:


loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y))


# In[22]:


optimizer = tf.train.AdamOptimizer(learning_rate=LEARN_RATE, name='Adam').minimize(loss)


# In[23]:


# Check to see if the prediction matches the label
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))


# In[24]:


# Calculate accuracy as mean of the correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[25]:


# TensorBoard data collection
tf.summary.scalar('cross_entropy_loss', loss)
tf.summary.scalar('accuracy', accuracy)
tf.summary.image('input_images', x)


# In[26]:


# set up saver object
saver = tf.train.Saver()


# In[27]:


# Launch the graph
with tf.Session() as sess:

    sess.run(tf.initializers.global_variables())
    
    # TensorBoard writer
    writer = tf.summary.FileWriter(TB_LOG_DIR, sess.graph)
    tb_summary = tf.summary.merge_all()

    # Training phase with training data
    print ('-------------------------------------------------------------')
    print ('TRAINING PHASE')
    print ('-------------------------------------------------------------')
    for epoch in range(EPOCHS):
        print ("Epoch", epoch+1, "/", EPOCHS)

        # process all batches
        for i in range(total_batches):
            
            # fetch a batch from training dataset
            batch_x, batch_y = x_train[i*BATCHSIZE:i*BATCHSIZE+BATCHSIZE], y_train[i*BATCHSIZE:i*BATCHSIZE+BATCHSIZE]

            # Run graph for optimization, loss, accuracy - i.e. do the training
            _, s = sess.run([optimizer, tb_summary], feed_dict={x: batch_x, y: batch_y})
            writer.add_summary(s, (epoch*total_batches + i))
            # Display accuracy per 100 batches
            if i % 100 == 0:
              acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
              print (' Step: {:4d}  Training accuracy: {:1.4f}'.format(i,acc))


    print ('-------------------------------------------------------------')
    print ('FINISHED TRAINING')
    print('Run `tensorboard --logdir=%s --port 6006 --host localhost` to see the results.' % TB_LOG_DIR)
    print ('-------------------------------------------------------------')
    writer.flush()
    writer.close()


    # Evaluation phase with test dataset
    print ('EVALUATION PHASE:')
    print ("Final Accuracy with validation set:", sess.run(accuracy, feed_dict={x: x_valid, y: y_valid}))
    print ('-------------------------------------------------------------')

    # save post-training checkpoint & graph
    print ('SAVING:')
    save_path = saver.save(sess, os.path.join(CHKPT_DIR, CHKPT_FILE))
    print('Saved checkpoint to %s' % os.path.join(CHKPT_DIR,CHKPT_FILE))
    tf.train.write_graph(sess.graph_def, CHKPT_DIR, TRAIN_GRAPH, as_text=False)
    print('Saved binary graphDef to %s' % os.path.join(CHKPT_DIR,TRAIN_GRAPH))
    print ('-------------------------------------------------------------')


#####  SESSION ENDS HERE #############


# In[32]:


saver.load_weights(CHKPT_PATH)


# In[ ]:




