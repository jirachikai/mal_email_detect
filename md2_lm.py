import tensorflow as tf
class LM(object):
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

    def __init__(self, model_config):
        self.learning_rate = tf.Variable(
            float(model_config.learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * model_config.learning_rate_decay_factor)
        self.keep_probability = tf.placeholder(tf.float32, name="keep_probability")

        self.input = tf.placeholder(tf.int64, shape=(model_config.batch_size, 
            model_config.time_steps), name="input")
        self.targets = tf.placeholder(tf.int64, shape=(model_config.batch_size, 
            model_config.time_steps), name="targets")
        self.init = tf.placeholder(tf.float32, shape=(), name="init") # init parameter 

        self.embedding = tf.Variable(tf.random_uniform(
            (model_config.vocabulary_size, model_config.hidden_units), 
            -self.init, self.init), dtype=tf.float32, name="embedding")
        self.embedded_input = tf.nn.embedding_lookup(self.embedding, 
            self.input, name="embedded_input")

        cell = tf.contrib.rnn.BasicLSTMCell(model_config.hidden_units)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_probability)
        rnn_layers = cell
        if model_config.num_layers > 1:
            rnn_layers = tf.contrib.rnn.MultiRNNCell([cell for _ in range(model_config.num_layers)])

        self.reset_state = rnn_layers.zero_state(model_config.batch_size, dtype=tf.float32)
        # print(self.reset_state)
        # self.state = tf.placeholder(tf.float32, self.reset_state.state_size, "h_state")
        # query = tf.concat(1,[state[0], state[1]])
        # self.sequence_length = tf.placeholder(tf.int64, ())
        self.outputs, self.next_state = tf.nn.dynamic_rnn(rnn_layers, self.embedded_input, time_major=True,
                                                            initial_state=self.reset_state)

        # Concatenate all the batches into a single row.
        self.flattened_outputs = tf.reshape(tf.concat(1, self.outputs), (-1, hidden_units),
                                            name="flattened_outputs")
        # Project the outputs onto the vocabulary.
        self.w = tf.get_variable("w", (hidden_units, 1))
        self.b = tf.get_variable("b", 1)
        self.predicted = tf.matmul(self.flattened_outputs, self.w) + self.b
        # Compare predictions to labels.
        self.loss = tf.nn.seq2seq.sequence_loss_by_example([self.predicted], [tf.concat(-1, self.targets)],
                [tf.ones(model_config.batch_size * model_config.time_steps)])
        self.cost = tf.div(tf.reduce_sum(self.loss), model_config.batch_size, name="cost")

        self.validation_perplexity = tf.Variable(dtype=tf.float32, initial_value=float("inf"), trainable=False,
                                                    name="validation_perplexity")
        tf.scalar_summary(self.validation_perplexity.op.name, self.validation_perplexity)
        self.training_epoch_perplexity = tf.Variable(dtype=tf.float32, initial_value=float("inf"), trainable=False,
                                                        name="training_epoch_perplexity")
        tf.scalar_summary(self.training_epoch_perplexity.op.name, self.training_epoch_perplexity)
        self.iteration = tf.Variable(0, dtype=tf.int64, name="iteration", trainable=False)
        self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()),
                                                    max_gradient, name="clip_gradients")
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_step = optimizer.apply_gradients(zip(self.gradients, tf.trainable_variables()),
                                                    name="train_step", global_step=self.iteration)

        self.initialize = tf.initialize_all_variables()
        self.summary = tf.merge_all_summaries()

    def train(self, session, init, training_set, parameters, exit_criteria, validation, logging_interval, directories):
        epoch = 1
        iteration = 0
        state = None
        summary = self.summary_writer(directories.summary, session)
        session.run(self.initialize, feed_dict={self.init: init})
        try:
            # Enumerate over the training set until exit criteria are met.
            while True:
                epoch_cost = epoch_iteration = 0
                # Enumerate over a single epoch of the training set.
                for start_document, context, target, complete in training_set.epoch(self.time_steps, self.batch_size):
                    if start_document:
                        state = session.run(self.reset_state)
                    _, cost, state, iteration = session.run(
                        [self.train_step, self.cost, self.next_state, self.iteration],
                        feed_dict={
                            self.input: context,
                            self.targets: target,
                            self.state: state,
                            self.learning_rate: parameters.learning_rate,
                            self.keep_probability: parameters.keep_probability
                        })
                    epoch_cost += cost
                    epoch_iteration += self.time_steps
                    if self._interval(iteration, logging_interval):
                        logger.info("Epoch %d (%0.4f complete), Iteration %d: epoch training perplexity %0.4f" %
                                    (epoch, complete, iteration, self.perplexity(epoch_cost, epoch_iteration)))
                    if validation is not None and self._interval(iteration, validation.interval):
                        validation_perplexity = self.test(session, validation.validation_set)
                        self.store_validation_perplexity(session, summary, iteration, validation_perplexity)
                        logger.info("Epoch %d, Iteration %d: validation perplexity %0.4f" %
                                    (epoch, iteration, validation_perplexity))
                    if exit_criteria.max_iterations is not None and iteration > exit_criteria.max_iterations:
                        raise StopTrainingException()

                self.store_training_epoch_perplexity(session, summary, iteration,
                                                     self.perplexity(epoch_cost, epoch_iteration))
                epoch += 1
                if exit_criteria.max_epochs is not None and epoch > exit_criteria.max_epochs:
                    raise StopTrainingException()
        except (StopTrainingException, KeyboardInterrupt):
            pass
        logger.info("Stop training at epoch %d, iteration %d" % (epoch, iteration))
        summary.close()
        if directories.model is not None:
            model_filename = self._model_file(directories.model)
            tf.train.Saver().save(session, model_filename)
            self._write_model_parameters(directories.model)
            logger.info("Saved model in %s " % directories.model)

    def _write_model_parameters(self, model_directory):
        parameters = {
            "max_gradient": self.max_gradient,
            "batch_size": self.batch_size,
            "time_steps": self.time_steps,
            "vocabulary_size": self.vocabulary_size,
            "hidden_units": self.hidden_units,
            "layers": self.layers
        }
        with open(self._parameters_file(model_directory), "w") as f:
            json.dump(parameters, f, indent=4)

    def test(self, session, test_set):
        state = None
        epoch_cost = epoch_iteration = 0
        for start_document, context, target, _ in test_set.epoch(self.time_steps, self.batch_size):
            if start_document:
                state = session.run(self.reset_state)
            cost, state = session.run([self.cost, self.next_state],
                                      feed_dict={
                                          self.input: context,
                                          self.targets: target,
                                          self.state: state,
                                          self.keep_probability: 1
                                      })
            epoch_cost += cost
            epoch_iteration += self.time_steps
        return self.perplexity(epoch_cost, epoch_iteration)

    @property
    def batch_size(self):
        return self.input.get_shape()[0].value

    @property
    def time_steps(self):
        return self.input.get_shape()[1].value

    @property
    def vocabulary_size(self):
        return self.embedding.get_shape()[0].value

    @property
    def hidden_units(self):
        return self.embedding.get_shape()[1].value

    @staticmethod
    def _interval(iteration, interval):
        return interval is not None and iteration > 1 and iteration % interval == 0

    @staticmethod
    def perplexity(cost, iterations):
        return np.exp(cost / iterations)

    def store_validation_perplexity(self, session, summary, iteration, validation_perplexity):
        session.run(self.validation_perplexity.assign(validation_perplexity))
        summary.add_summary(session.run(self.summary), global_step=iteration)

    def store_training_epoch_perplexity(self, session, summary, iteration, training_perplexity):
        session.run(self.training_epoch_perplexity.assign(training_perplexity))
        summary.add_summary(session.run(self.summary), global_step=iteration)

    @staticmethod
    def summary_writer(summary_directory, session):
        class NullSummaryWriter(object):
            def add_summary(self, *args, **kwargs):
                pass

            def flush(self):
                pass

            def close(self):
                pass

        if summary_directory is not None:
            return tf.train.SummaryWriter(summary_directory, session.graph)
        else:
            return NullSummaryWriter()

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
    hidden_units,
    num_layers,
    init = 0.1,
    model = None,
    summary = None,
    learning_rate = 1.0,
    learning_rate_decay_factor = 0.8,
    keep_probability = 0.8
    ):
        # Path Related:
        self.model = model
        self.summary = summary

        # Parameters:
        self.init = init # init value range of parameter
        self.learning_rate = learning_rate
        self.keep_probability = keep_probability
        self.batch_size = batch_size
        self.time_steps =  time_steps
        self.vocabulary_size = vocabulary_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.learning_rate_decay_factor = learning_rate_decay_factor

        # Model Setting:

