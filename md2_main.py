import os

from md_data_utils import *
from md2_lm import *

max_len = 199
voc_path = "email_data_test/voc"
data_path = "email_data_test/test.csv"
buckets = [(max_len, 1)]

def dataload():
    new_bucketed_data, voc_len, max_email_len = read_bucketed_data(data_path, buckets, voc_path)
    return new_bucketed_data, voc_len, max_email_len

def create_model(session, voc_len, max_email_len, forward_only = False):
    mc = Model_Conifg(
        batch_size = 50,
        time_steps = max_email_len,
        vocabulary_size = voc_len,
        hidden_units = 80,
        num_layers = 1,
        max_gradient = 10
    )
    model = LM(mc)
    session.run(model.initialize, feed_dict={model.init: mc.init})

#   ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
#   if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
#     print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
#     model.saver.restore(session, ckpt.model_checkpoint_path)
#   else:
#     print("Created model with fresh parameters.")
#     session.run(tf.global_variables_initializer())    
    return model


def train(new_bucketed_data, Model):
    model_directory = ""
    # validation =  Validation(args.validation_interval, args.data_set[args.validation_partition])

    # Run training.
    start_time = time.time()
    with tf.Graph().as_default():
        mc = Model_Conifg(
            batch_size = 50,
            time_steps = max_email_len,
            vocabulary_size = voc_len,
            hidden_units = 80,
            num_layers = 1,
            max_gradient = 10
        )
        model = LM(max_gradient, batch_size, time_steps, len(args.data_set.vocabulary),
                    args.hidden_units, args.layers)

    #     with tf.Session() as session:
    #         model.train(session,
    #                     init,
    #                     data_set[training_partition],
    #                     Parameters(learning_rate, keep_probability),
    #                     ExitCriteria(max_iterations, max_epochs),
    #                     validation,
    #                     logging_interval,
    #                     Directories(model_directory, summary_directory))
    # logger.info("Total training time %s" % timedelta(seconds=(time.time() - start_time)))

new_bucketed_data, voc_len, max_email_len = dataload()
print("voc_len: %d",(voc_len))
print("max_email_len: %d",(max_email_len))
with tf.Session() as session:
    model = create_model(session, voc_len, max_email_len)
# create_model(voc_len, max_email_len)
