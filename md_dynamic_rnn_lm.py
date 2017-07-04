import tensorflow as tf
import random
import numpy as np
import logging
import os
import json

class dynamicLM(object):
    def __init__(self, model_config):
        self.learning_rate = tf.Variable(
            float(model_config.learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * model_config.learning_rate_decay_factor)

        self.x = tf.placeholder(tf.int32, [None, model_config.time_steps], name = "x")
        self.y = tf.placeholder(tf.float32, [None, model_config.n_classes], name = "y")
        self.x_lengths = tf.placeholder(tf.int32, [None], name = "x_lengths")

        self.embedding = tf.Variable(tf.random_uniform(
            (model_config.vocabulary_size, model_config.hidden_units), 
            -model_config.init, model_config.init), dtype=tf.float32, name="embedding")
        self.embedded_input = tf.nn.embedding_lookup(self.embedding, 
            self.x, name="embedded_input")

        cell = tf.contrib.rnn.BasicLSTMCell(model_config.hidden_units)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=model_config.keep_probability)
        rnn_layers = cell
        if model_config.num_layers > 1:
            rnn_layers = tf.contrib.rnn.MultiRNNCell([cell for _ in range(model_config.num_layers)])

        self.reset_state = rnn_layers.zero_state(model_config.batch_size, dtype=tf.float32)
        self.outputs, self.last_states = tf.nn.dynamic_rnn(cell = rnn_layers, inputs = self.embedded_input, 
            sequence_length = self.x_lengths, time_major=False, initial_state = self.reset_state)
        # self.last_states = tf.unstack(tf.transpose(self.outputs, perm = [1,0,2]))[-1]
        self.last_states = self.last_states[1] # 0 is cell, 1 is hidden

        # Project the outputs onto the vocabulary.
        self.W = tf.get_variable("w", (model_config.hidden_units, model_config.n_classes))
        self.b = tf.get_variable("b", model_config.n_classes)
        self.pred = tf.matmul(self.last_states, self.W) + self.b

        # Compare predictions to labels.
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.pred, labels = self.y))

        self.validation_perplexity = tf.Variable(dtype=tf.float32, initial_value=float("inf"), trainable=False,
                                                    name="validation_perplexity")
        self.training_epoch_perplexity = tf.Variable(dtype=tf.float32, initial_value=float("inf"), trainable=False,
                                                        name="training_epoch_perplexity")

        self.iteration = tf.Variable(0, dtype=tf.int64, name="iteration", trainable=False)
        self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()),
                            model_config.max_gradient, name="clip_gradients")
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_step = optimizer.apply_gradients(zip(self.gradients, tf.trainable_variables()),
                                                    name="train_step", global_step=self.iteration)
        
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.initialize = tf.initialize_all_variables()
        self.model_config = model_config
        self.ckpt_num = 0

    def step(self, session, batch_x, batch_y, batch_x_lengths, forward_only = False):
        batch_x = batch_x.reshape([self.model_config.batch_size, self.model_config.time_steps])
        if forward_only: # predict
            batch_loss, batch_accuracy = session.run(
                [self.cost, self.accuracy], 
                feed_dict={
                    self.x_lengths: batch_x_lengths,
                    self.x: batch_x,
                    self.y: batch_y,
                })
            return batch_loss, batch_accuracy, 
        else: # training
            _, batch_loss, batch_accuracy = session.run(
                [self.train_step, self.cost, self.accuracy], 
                feed_dict={
                    self.x_lengths: batch_x_lengths,
                    self.x: batch_x,
                    self.y: batch_y,
                })
            return batch_loss, batch_accuracy

    def train(self, session, training_set, validation, exit_criteria, batch_per_epoch, 
                directories, ckpt_per_epoch, restore_env = None):
        logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=self._log_file(directories),
                    filemode='w')
        logger = logging.getLogger("train")
        session.run(self.initialize)
        self._write_model_parameters(directories)
        logger.info("Save Model Parameters Successfully!")
        print("Save Model Parameters Successfully!")
        if restore_env == None:
            session.run(self.initialize)
            epoch = 0
        else:
            epoch = restore_env.epoch

        iteration = 0
        epoch = 0
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        if validation:
            validation_X, validation_Y, validation_len = self.padding(validation, self.model_config.time_steps)

        # epoch contains batch_per_epoch batches
        while True:
            batch_x, batch_y, batch_x_lengths = self.get_batch(training_set)
            # print(batch_x)
            batch_loss, batch_accuracy = self.step(session, batch_x, 
                batch_y, batch_x_lengths, False)
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
            print("Epoch %d, Iteration %d: batch loss %0.4f, batch accuracy %0.4f"
                 % (epoch, iteration, batch_loss, batch_accuracy))
            if not iteration%batch_per_epoch:
                epoch += 1 
                epoch_accuracy /= float(batch_per_epoch)
                epoch_loss /= float(batch_per_epoch)
                logger.info("Epoch %d, Iteration %d: epoch loss %0.4f, epoch accuracy %0.4f"
                 % (epoch, iteration, epoch_loss, epoch_accuracy))
                print("Epoch %d, Iteration %d: epoch loss %0.4f, epoch accuracy %0.4f"
                 % (epoch, iteration, epoch_loss, epoch_accuracy))
                epoch_loss = 0.0
                epoch_accuracy = 0.0

                if validation:
                    validation_accuracy = 0.0
                    validation_loss = 0.0
                    t = int(validation_Y.shape[0]/self.model_config.batch_size)
                    for i in range(t):
                        validation_X_batch = validation_X[i*self.model_config.batch_size:(i+1)*self.model_config.batch_size]
                        validation_Y_batch = validation_Y[i*self.model_config.batch_size:(i+1)*self.model_config.batch_size]
                        validation_len_batch = validation_len[i*self.model_config.batch_size:(i+1)*self.model_config.batch_size]

                        validation_loss_batch, validation_accuracy_batch = \
                            self.step(session, validation_X_batch, validation_Y_batch, validation_len_batch, True)
                        validation_accuracy += validation_accuracy_batch
                        validation_loss += validation_loss_batch

                    validation_accuracy /= t
                    validation_loss /= t
                    logger.info("Epoch %d, Iteration %d: validation accuracy %0.4f, validation loss %0.4f" %
                                (epoch, iteration, validation_accuracy, validation_loss))
                    print("Epoch %d, Iteration %d: validation accuracy %0.4f, validation loss %0.4f" %
                                (epoch, iteration, validation_accuracy, validation_loss))
                    

            if epoch > exit_criteria.max_epochs:
                break
            iteration += 1

        logger.info("Stop training at epoch %d, iteration %d" % (epoch, iteration))

        model_filename = self._model_file(directories)
        tf.train.Saver().save(session, model_filename)
        
        logger.info("Saved model in %s " % directories)

    @staticmethod
    def padding(data, times_steps, PAD_ID = 0):
        padded_xs = []
        ys = []
        xs_length = []
        for x, y in data:
            if len(x) > times_steps:
                print("len(x) > times_steps, len(x) = %d"%(len(x)))
                print(x)
            pad_size = times_steps - len(x)
            xs_length.append(len(x))
            padded_xs.append(x + [PAD_ID] * pad_size)
            if int(y) == 1:
                ys.append([0,1])
            else:
                ys.append([1,0])
        return np.array(padded_xs), np.array(ys), np.array(xs_length)

    def get_batch(self, data):
        batch_data = []
        # Get a random batch of inputs from data,
        for _ in range(self.model_config.batch_size - 1):
            batch_data.append(random.choice(data[0]))
        batch_data.append(random.choice(data[1]))
        # pad them if needed
        # will return padded x, y and x origin length
        return self.padding(batch_data, self.model_config.time_steps)

    @classmethod
    def restore(cls, session, model_directory):
        """
        Restore a previously trained model

        :param session: session into which to restore the model
        :type session: TensorFlow Session
        :param model_directory: directory to which the model was saved
        :type model_directory: str
        :return: trained model
        :rtype: RNN
        """
        with open(cls._parameters_file(model_directory)) as f:
            parameters = json.load(f)
        model = cls(parameters["max_gradient"],
                    parameters["batch_size"], parameters["time_steps"], parameters["vocabulary_size"],
                    parameters["hidden_units"], parameters["layers"])
        tf.train.Saver().restore(session, cls._model_file(model_directory))
        return model

    @staticmethod
    def _parameters_file(model_directory):
        return os.path.join(model_directory, "parameters.json")
    @staticmethod
    def _model_file(model_directory):
        return os.path.join(model_directory, "model")
    @staticmethod
    def _log_file(model_directory):
        return os.path.join(model_directory, "logging.log")
  
    def _write_model_parameters(self, model_directory):
        with open(self._parameters_file(model_directory), "w") as f:
            json.dump(self.model_config.__dict__, f, indent=4)

# Objects used to group training parameters
class ExitCriteria(object):
    def __init__(self, max_iterations, max_epochs):
        self.max_iterations = max_iterations
        self.max_epochs = max_epochs

class Validation(object):
    def __init__(self, interval, validation_set):
        self.interval = interval
        self.validation_set = validation_set

class Model_Conifg(object):
    def __init__(self,
    batch_size,
    time_steps,
    vocabulary_size,
    n_classes,
    hidden_units,
    num_layers,
    max_gradient,
    init = 0.1,
    model = None,
    learning_rate = 1.0,
    learning_rate_decay_factor = 0.8,
    keep_probability = 0.8
    ):
        # Path Related:
        self.model = model

        # Parameters:
        self.init = init # init value range of parameter
        self.learning_rate = learning_rate
        self.keep_probability = keep_probability
        self.batch_size = batch_size
        self.time_steps =  time_steps
        self.vocabulary_size = vocabulary_size
        self.n_classes = n_classes
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.max_gradient = max_gradient

        # Model Setting:

