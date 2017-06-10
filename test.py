from md_data_utils import *


def test_read_data():
    buckets = [(5, 1), (10, 1), (50, 1)]
    data = read_bucketed_data(
        "email_data_test/test.csv", buckets, "email_data_test/voc")
    return str(data)



print(test_read_data())
