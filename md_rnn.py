from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils


class Seq2SeqModel(object):
    def __init__(self,
                 input_size,
                 tag_size,
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 use_lstm=False,
                 use_att=False,
                 forward_only=False,
                 dtype=tf.float32):
        """Create the model.

        Args:
          source_vocab_size: size of the source vocabulary.
          tag_size: size of the tag.
          buckets: a list of pairs (I, O), where I specifies maximum input length
            that will be processed in that bucket, and O specifies maximum output
            length. Training instances that have inputs longer than I or outputs
            longer than O will be pushed to the next bucket and padded accordingly.
            We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
          size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          num_samples: number of samples for sampled softmax.
          forward_only: if set, we do not construct the backward pass in the model.
          dtype: the data type to use to store internal variables.
        """
        self.input_size = input_size
        self.tag_size = tag_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # Create the internal multi-layer cell for our RNN.
        def single_cell():
            return tf.contrib.rnn.GRUCell(size)
        if use_lstm:
            def single_cell():
                return tf.contrib.rnn.BasicLSTMCell(size)
        cell = single_cell()
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell(
                [single_cell() for _ in range(num_layers)])

        def seq_model(encoder_inputs, decoder_inputs, do_decode):
            return tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols=input_size,
                num_decoder_symbols=tag_size,
                embedding_size=size,
                feed_previous=do_decode,
                dtype=dtype)
        if use_att:
            def seq_model(encoder_inputs, decoder_inputs, do_decode):
                return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs,
                    decoder_inputs,
                    cell,
                    num_encoder_symbols=input_size,
                    num_decoder_symbols=tag_size,
                    embedding_size=size,
                    feed_previous=do_decode,
                    dtype=dtype)

        # Feeds for inputs.
        self.inputs = []
        self.tag = []
        self.target_weight = []

        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.inputs.append(tf.placeholder(tf.int32, shape=[None],
                                              name="input{0}".format(i)))

        self.tag.append(tf.placeholder(tf.int32, shape=[None], name='tag'))
        self.target_weight.append(tf.placeholder(
            dtype, shape=[None], name="target_weigt"))
        target = [self.tag[0]]

        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.inputs, self.tag, target,
                self.target_weight, buckets, lambda x, y: seq_model(x, y, True))
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.inputs, self.tag, target,
                self.target_weight, buckets, lambda x, y: seq_model(x, y, False))

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, inputs, tags, target_weights,
             bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          inputs: list of numpy int vectors to feed as inputs.
          tags: list of numpy int to feed as tags.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.

        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        input_size, tag_size = self.buckets[bucket_id]
        if len(inputs) != input_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(tags) != tag_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(tag_size), decoder_size))
        if len(target_weights) != tag_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(input_size):
            input_feed[self.inputs[l].name] = inputs[l]
        for l in xrange(tag_size):
            input_feed[self.tag[l].name] = tags[l]
            input_feed[self.target_weight[l].name] = target_weights[l]

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(tag_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            # Gradient norm, loss, no outputs.
            return outputs[1], outputs[2], None
        else:
            # No gradient norm, loss, outputs.
            return None, outputs[0], outputs[1:]

    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.

        Returns:
          The tuple (batch_inputs, tags) for
          the constructed batch that has the proper format to call step(...) later.
        """
        seq_max_len = self.buckets[bucket_id][0]
        input_vecs = []
        tags = []

        # Get a random batch of inputs from data,
        # pad them if needed
        for _ in xrange(self.batch_size):
            input_vec, tag = random.choice(data[bucket_id])

            # Input are padded then.
            input_vec_pad_size = seq_max_len - len(input_vec) - 1
            input_vecs.append([data_utils.GO_ID] + decoder_input +
                              [data_utils.PAD_ID] * decoder_pad_size)
            tags.append(tag)

        # Now we create batch-major vectors from the data selected above.
        batch_inputs = []

        # Batch inputs are just re-indexed inputs.
        # size of batch_inputs: seq_max_len * batch_size
        for length_idx in xrange(seq_max_len):
            batch_inputs.append(
                np.array([input_vecs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))
        batch_tags = np.array(tags)
        batch_weights = np.array([[1] for _ in batch_tags])
        return batch_inputs, batch_tags, batch_weights
