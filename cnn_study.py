import tensorflow as tf
import numpy as np
import pandas as pd

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    conv1 = tf.nn.conv2d(X, w,strides=[1, 1, 1, 1],\
                         padding='SAME')
    conv1_a = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1_a, ksize=[1, 2, 2, 1]\
                       ,strides=[1, 2, 2, 1],\
                       padding='SAME')
    conv1 = tf.nn.dropout(conv1, p_keep_conv)
    conv2 = tf.nn.conv2d(conv1, w2,\
                     strides=[1, 1, 1, 1],\
                     padding='SAME')
    conv2_a = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],\
                        strides=[1, 2, 2, 1],\
                        padding='SAME')
    conv2 = tf.nn.dropout(conv2, p_keep_conv)

    conv3=tf.nn.conv2d(conv2, w3,\
                       strides=[1, 1, 1, 1]\
                       ,padding='SAME')

    conv3 = tf.nn.relu(conv3)

    FC_layer = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],\
                    strides=[1, 2, 2, 1],\
                    padding='SAME')

    FC_layer = tf.reshape(FC_layer,\
                          [-1, w4.get_shape().as_list()[0]])
    FC_layer = tf.nn.dropout(FC_layer, p_keep_conv)

    output_layer = tf.nn.relu(tf.matmul(FC_layer, w4))
    output_layer = tf.nn.dropout(output_layer, p_keep_hidden)

    result = tf.matmul(output_layer, w_o)
    return result

batch_size = 100 #reduce if RAM is leaking
test_size = 20
img_size = 28
num_channel = (6+1)*6/2*2
num_classes = 20

FEM_parm = pd.read_csv('FEM_parm_frf.csv', header=None, names=np.arange(1, 22))
test_parm = pd.read_csv('test_parm_frf.csv', header=None, names=np.arange(1, 22))
# Drop the first parameter
FEM_parm=FEM_parm.drop([1],axis=1)
test_parm=test_parm.drop([1],axis=1)
FEM_freq = pd.read_csv('FEM_frf.csv', header=None)
test_freq = pd.read_csv('test_frf.csv', header=None)
# Convert the string to complex
FEM_freq = FEM_freq.apply(lambda col: col.apply(lambda val: complex(val.strip('()'))))
test_freq = test_freq.apply(lambda col: col.apply(lambda val: complex(val.strip('()'))))
# sns.jointplot(x=FEM_parm[1],y=FEM_parm[2])
# #Cut the input to bins
trus = 0.15
bins = [0, (1-trus)*7e10, (1+trus)*7e10, 3*7e10]
FEM_parm_cats = pd.DataFrame()
test_parm_cats = pd.DataFrame()
for col in FEM_parm:
    FEM_parm_cats[col] = pd.cut(FEM_parm[col], bins, labels=[
                                'lower', 'in', 'higher']).cat.codes
for col in test_parm:
    test_parm_cats[col] = pd.cut(test_parm[col], bins, labels=[
                                 'lower', 'in', 'higher']).cat.codes

new_X = np.concatenate((FEM_freq.values.real,FEM_freq.values.imag),axis=1)
trX = np.ascontiguousarray(new_X, dtype=np.float32)
# test_X = ((test_freq.values/mean_test_freq-1)*10)
new_test_X = np.concatenate((test_freq.values.real,test_freq.values.imag),axis=1)
teX = np.ascontiguousarray(new_test_X, dtype=np.float32)
y = FEM_parm_cats.values
trY = np.ascontiguousarray(y, dtype=np.int8)
test_y = test_parm_cats.values
teY = np.ascontiguousarray(test_y, dtype=np.int8)

img_size_x = int(num_channel)
img_size_y = int(trX.shape[1]/num_channel)

trX = trX.reshape(-1, img_size_x, img_size_y, 1)
teX = teX.reshape(-1, img_size_x, img_size_y, 1)
#%% End of loading data

X = tf.placeholder("float", [None, img_size_x, img_size_y, 1])
Y = tf.placeholder("float", [None, num_classes])

w = init_weights([2, 100, 1, 32])
w2 = init_weights([2, 100, 32, 64])
w3 = init_weights([2, 100, 64, 128])
w4 = init_weights([115200, 625])
w_o = init_weights([625, num_classes])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)
Y_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=py_x)

cost = tf.reduce_mean(Y_)

optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

#%% Start calculation

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(10):
        training_batch = \
                       zip(range(0, len(trX), \
                                 batch_size),
                             range(batch_size, \
                                   len(trX)+1, \
                                   batch_size))
        for start, end in training_batch:
            sess.run(optimizer , feed_dict={X: trX[start:end],\
                                          Y: trY[start:end],\
                                          p_keep_conv: 0.8,\
                                          p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==\
                         sess.run\
                         (predict_op,\
                          feed_dict={X: teX[test_indices],\
                                     Y: teY[test_indices], \
                                     p_keep_conv: 1.0,\
                                     p_keep_hidden: 1.0})))
