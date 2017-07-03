import tensorflow as tf

lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
embedding = tf.Variable(tf.random_uniform((vocabulary_size, hidden_units), -self.init, self.init),
                            dtype=tf.float32, name="embedding")
embedded_input = tf.nn.embedding_lookup(embedding, x, name="embedded_input")
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    'out': tf.Variable(tf.constant(0.1, shape = [n_classes]))
}

def dynamic_RNN():
    cell = tf.nn.rnn_cell.LSTMCell(hidden_units)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_probability)
    rnn_layers = cell
    if layers > 1:
        rnn_layers = tf.nn.rnn_cell.MultiRNNCell([cell for _ in range(num_layers))
    self.reset_state = rnn_layers.zero_state(batch_size, dtype=tf.float32)
    self.state = tf.placeholder(tf.float32, self.reset_state.get_shape(), "state")
    # self.sequence_length = tf.placeholder(tf.int64, ())
    self.outputs, self.next_state = tf.nn.dynamic_rnn(rnn_layers, self.embedded_input, time_major=True,
                                                        initial_state=self.state)


def RNN(X, weights, biases):
    # hidden layer
    # X: batch * 28 steps * 28 inputs
    #  ==> (128*28,28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # X_in ==> (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # lstm state: c_state (cell), m_state(hidden)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # will return batch_size * state_size
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # outputs: The RNN output 'Tensor'. shape: [batch_size, max_time, cell.output_size]
    # state: The final state. 
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state = _init_state, time_major = False)

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
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
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
