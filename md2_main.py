import os

from md_data_utils import *
from md2_lm import *
from md_dynamic_rnn_lm import *

voc_path = "swm_train_log/voc"
training_data_path_pos = "data/swm_training_pos.csv"
training_data_path_neg = "data/swm_training_neg.csv"
# training_data_path = "data/small_training_set.csv"
validation_data_path = "data/swm_validation.csv"
working_dir = "swm_train_log/"

#for testing
# voc_path = "email_data_test/voc"
# data_path = "email_data_test/test.csv"
# working_dir = "email_data_test/"

ec = ExitCriteria(max_iterations = None, max_epochs = 1000)
batch_per_epoch = 1000
ckpt_per_epoch = 5
max_size = 10000000
# ec = ExitCriteria(max_iterations = None, max_epochs = 5)
# batch_per_epoch = 2
# ckpt_per_epoch = 1

training_pos, voc_len, max_email_len_pos = read_data(training_data_path_pos, voc_path, create_voc = False, max_size = max_size)
training_neg, voc_len, max_email_len_neg = read_data(training_data_path_neg, voc_path, create_voc = False, max_size = max_size)
validation, voc_len, max_email_len = read_data(validation_data_path, voc_path, create_voc = False, max_size = max_size)
max_email_len = max(max_email_len_pos, max_email_len_neg, max_email_len)

dataset = [training_neg, training_pos]
print("voc_len: %d",(voc_len))
print("max_email_len: %d",(max_email_len))

create_voc = True,

mc = Model_Conifg(
    batch_size = 1000,
    # batch_size = 10,
    time_steps = max_email_len,
    vocabulary_size = voc_len,
    hidden_units = 80,
    num_layers = 1,
    max_gradient = 10,
    n_classes = 2
)

def create_dynamic_rnn_model(session, mc, forward_only = False):
    model = dynamicLM(mc)
    return model   

def create_rnn_model(session, mc, forward_only = False):
    model = LM(mc)
    return model   

with tf.Session() as session:
    model = create_dynamic_rnn_model(session, mc)
    # session, training_set, validation, exit_criteria, batch_per_epoch, directories, ckpt_per_epoch
    model.train(session = session, training_set = dataset, validation = validation,
         exit_criteria = ec, batch_per_epoch = batch_per_epoch, 
         directories = working_dir, ckpt_per_epoch = ckpt_per_epoch)
print("finished!")
