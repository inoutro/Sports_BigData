import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def reset_graph():
    tf.reset_default_graph()


def max_norm_regularizer(threshold, axes=1, name="max_norm",
                         collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None  # there is no regularization loss term

    return max_norm



data_sam = pd.read_csv("../input_data/final_dataset/sam_data_180214.csv")

X_col = ["chulNo" , "age" , "wgBudam"  , "rank" , "rcCntY" , "ord1CntY" , "ord2CntY" , "chaksunT" ,
         "chaksunY" , "chaksun_6m" , "trAge" , "rcCntY_tr" , "ord1CntY_tr" , "ord2CntY_tr" , 'age_jk' ,
         "wg_Budam_jk" , "rcCntY_jk" , "ord1CntY_jk" , "ord2CntY_jk" , "num_own" , "num_reg" , "num_cancel" ,
         "chaksunT_ow" , "chaksunY_ow" , "ord1CntY_ow" , "ord2CntY_ow" , "ord3CntY_ow"  , "rcCntY_ow" ,
         "hr_ord1PerT" , "hr_ord2PerT" , "tr_ord1PerT" , "tr_ord2PerT" , "jk_ord1PerT" , "jk_ord2PerT" ,
         "ow_ord1PerT" , "ow_ord2PerT" , "ow_ord3PerT" , "tr_career" , "jk_career" , "ow_career", "sex_female",
         "sex_gelding", "sex_male", "name_domestic", "name_foreign"]

y_col = ["ord"]



print(len(X_col))


## model

reset_graph()

n_inputs = 45     # input data variables node
n_hidden1 = 90     # hidden layer 1 node
n_hidden2 = 90      # hidden layer 2 node
n_hidden3 = 90      # hidden layer 2 node
n_hidden4 = 90      # hidden layer 2 node
n_outputs = 2        # output node: output class is 4 (LH / RH / AD / VS)

learning_rate = 0.01 # learning rate
dropout_rate = 0.1   # dropout rate

n_epochs = 100        # epoch
batch_size = 100     # batch size


# input, output placeholder setting
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")


# Dropout
training = tf.placeholder_with_default(False, shape=(), name='training')
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

# Max-Norm Regularization
max_norm_reg = max_norm_regularizer(threshold=1.0)


# Deep neural network design
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X_drop, n_hidden1, name="hidden1",
                              activation=tf.nn.selu, kernel_regularizer=max_norm_reg)
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, name="hidden2",
                              activation=tf.sigmoid, kernel_regularizer=max_norm_reg)
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    # hidden3 = tf.layers.dense(hidden2_drop, n_hidden3, name="hidden3",
    #                           activation=tf.nn.relu, kernel_regularizer=max_norm_reg)
    # hidden3_drop = tf.layers.dropout(hidden3, dropout_rate, training=training)
    logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    # Computes sparse softmax cross entropy between logits and labels
    loss = tf.reduce_mean(xentropy, name="loss")
    # Computes the mean of elements across "xentropy"

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # I implement Gradient Descent Optimizer
    training_op = optimizer.minimize(loss)
    # minimize loss to train

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    # Computes accuracy

init = tf.global_variables_initializer()
saver = tf.train.Saver()


# run the weights clipping operations after each training operation
clip_all_weights = tf.get_collection("max_norm")


#
kf = KFold(n_splits=5, shuffle=True)
for TR, TE in kf.split(np.unique(data_sam["race"])):
    TR, TE = TR + 1, TE + 1
    # print(TR)
    # print(len(TR))
    # print(TE)
    # print(len(TE))
    train = data_sam[data_sam["race"].isin(TR)]
    test = data_sam[data_sam["race"].isin(TE)]

    X_train = pd.DataFrame.as_matrix(train[X_col])
    y_train = pd.DataFrame.as_matrix(train[y_col])
    X_test = pd.DataFrame.as_matrix(test[X_col])
    y_test = pd.DataFrame.as_matrix(test[y_col])

    with tf.Session() as sess:
        init.run()
        accuracy_test = []
        for epoch in range(n_epochs):
            # Slicing train data by batch size(30)
            i = 0
            for batch in range(len(X_train) // batch_size):
                X_batch = X_train[i:i + batch_size]
                Y_batch = y_train[i:i + batch_size]
                i += batch_size
                # Training the weights, biases - Deep Neural Networks
                a = sess.run(training_op, feed_dict={X: X_batch, y: Y_batch.reshape(-1)})
                # print(a)
            # Computes accuracy training set





            # acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train.reshape(-1)})
            # Computes accuracy test set
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test.reshape(-1)})
            # append the test set accuracy in epoch
            accuracy_test.append(acc_test)
            # print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
            print("Test accuracy:", acc_test)
            print(accuracy_test)
            # race_accuracy_df = pd.DataFrame()

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))