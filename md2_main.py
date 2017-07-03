import os

from md_data_utils import *
from md2_lm import *
from md_dynamic_rnn import *

max_len = 199
voc_path = "email_data_test/voc"
data_path = "email_data_test/test.csv"
buckets = [(max_len, 1)]

def dataload():
    new_bucketed_data, voc_len, max_email_len = read_bucketed_data(data_path, buckets, voc_path)
    return new_bucketed_data, voc_len, max_email_len

def create_dynamic_rnn_model(session, mc, forward_only = False):
    model = dynamicLM(mc)
    return model   

def create_rnn_model(session, mc, forward_only = False):
    model = LM(mc)
    return model   

ec = ExitCriteria(max_iterations = 1000, max_epochs = 5)
new_bucketed_data, voc_len, max_email_len = dataload()
mc = Model_Conifg(
    batch_size = 50,
    time_steps = max_email_len,
    vocabulary_size = voc_len,
    hidden_units = 80,
    num_layers = 1,
    max_gradient = 10,
    n_classes = 2
)
print("voc_len: %d",(voc_len))
print("max_email_len: %d",(max_email_len))
with tf.Session() as session:
    model = create_rnn_model(session, mc)
    model.train(session, new_bucketed_data[0], None, ec, 5,"email_data_test/", 2)
print("finished!")
