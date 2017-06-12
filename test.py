import tensorflow as tf
from md_data_utils import *
lr = 0.001
training_iters = 100000
batch_size = 128
folder = 'email_data_test/'
# vocabulary_size = 120
# max_email_len = 199
n_hidden_units = 90
n_classes = 2
buckets = [(199, 1)]

new_bucketed_data, vocabulary_size, max_email_len = read_bucketed_data(
            folder + "test.csv", buckets, folder + "voc")

x = tf.placeholder(tf.int32, [None, max_email_len])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    'out': tf.Variable(tf.constant(0.1, shape = [n_classes]))
}

def RNN(X, weights, biases):
    embedding = tf.Variable(tf.random_uniform(
        (vocabulary_size, n_hidden_units), -0.1, 0.1), 
        dtype=tf.float32, name="embedding")
    embedded_input = tf.nn.embedding_lookup(embedding, X, name="embedded_input")

    print(embedded_input.shape)

    # lstm state: c_state (cell), m_state(hidden)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # will return batch_size * state_size
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # outputs: The RNN output 'Tensor'. shape: [batch_size, max_time, cell.output_size]
    # state: The final state. 
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, embedded_input, initial_state = _init_state, time_major = False)

    # hidden layer for output
    # state[1]: last m_state(state[0]: c_state) or use
    # permutation: [max_time, batch_size, cell.output_size], so -1 is the last state
    outputs = tf.unstack(tf.transpose(outputs, perm = [1,0,2])) 
    results = tf.matmul(outputs[-1], weights['out']) + biases['out'] # shape = (128, 10)

    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
step = 0

while step*batch_size < training_iters:
    batch_xs, batch_ys = get_batch(batch_size, new_bucketed_data, 0, max_email_len)
    batch_xs = batch_xs.reshape([batch_size, max_email_len])
    sess.run([train_op], feed_dict={
        x: batch_xs,
        y: batch_ys
    })
    if step%20 == 0:
        print(sess.run(accuracy,feed_dict={
            x: batch_xs,
            y: batch_ys
        }))
    step+=1
