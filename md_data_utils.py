from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def basic_tokenizer(email):
    """Very basic tokenizer: split the email into a list of tokens."""
    return list(email)


def create_vocabulary(vocabulary_path, bucketed_data, tokenizer=None):
    """Create vocabulary file (if it does not exist yet) from data file.

    Args:
    vocabulary_path: path where the vocabulary will be created.
    bucketed_data: bucketed data that will be used to create vocabulary.
    tokenizer: a function to use to tokenize each data email;
    if None, basic_tokenizer will be used.
    """
    print("Creating vocabulary %s" % (vocabulary_path))
    vocab = {}
    counter = 0
    for bucket in bucketed_data:
        for (email, _) in bucket:
            counter += 1
            if counter % 100000 == 0:
                print("  processing line %d" % counter)
            tokens = tokenizer(
                email) if tokenizer else basic_tokenizer(email)
            for w in tokens:
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
    vocab_list = _START_VOCAB + \
                sorted(vocab, key=vocab.get, reverse=True)
    with open(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
            vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.
    Args:
    vocabulary_path: path to the file containing the vocabulary.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with open(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def email_to_token_ids(email, vocabulary, tokenizer=None):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
    email: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
    if None, basic_tokenizer will be used.

    Returns:
    a list of integers, the token-ids for the email.
    """
    if tokenizer:
        words = tokenizer(email)
    else:
        words = basic_tokenizer(email)
    return [vocabulary.get(str.encode(w), UNK_ID) for w in words]

def data_to_token_ids(bucketed_data, vocabulary, tokenizer=None):
    new_bucketed_data = []
    for bucket in bucketed_data:
        new_bucket = []
        for (email, tag) in bucket:
            new_bucket.append((email_to_token_ids(email, vocabulary), tag))
        new_bucketed_data.append(new_bucket)
    return new_bucketed_data


def read_bukcketed_data_raw(fp, buckets, max_size=None):
    data_set = [[] for _ in buckets]
    with open(fp, "r") as data:
        ins = data.readline().strip()
        counter = 0
        while ins and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            email, tag = ins.split('\x07')
            # need to change email to vector
            for bucket_id, (email_size, _) in enumerate(buckets):
                if len(email) < email_size:
                    data_set[bucket_id].append((email, tag))
                    break
            ins = data.readline().strip()
    return data_set


def read_bucketed_data(fp, buckets, vocabulary_path, max_size=None):
    bucketed_data_raw = read_bukcketed_data_raw(fp, buckets)
    create_vocabulary(vocabulary_path, bucketed_data_raw)
    vocab, rev_vocab = initialize_vocabulary(vocabulary_path)
    new_bucketed_data = data_to_token_ids(bucketed_data_raw, vocab)
    return new_bucketed_data

if __name__ == '__main__':
    def test_read_data():
        folder = 'email_data_test/'
        buckets = [(5, 1), (10, 1), (50, 1)]
        data = read_bucketed_data(
            folder + "test.csv", buckets, folder + "voc")
        return str(data)
    print(test_read_data())
    # print(test_read_data())
    print("finish!")