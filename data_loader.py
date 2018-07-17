import codecs
import numpy as np

def read_data(data_path_pos, data_path_neg, max_sequence_length):
    x = list()
    input_tags = list()
    y = list()

    with codecs.open(data_path_pos, 'r') as f1:
        for line in f1.readlines():
            line = line.strip()
            words, tags = line.split("#")
            words = words.strip().split()
            tags = tags.strip().split()
            words = words[:max_sequence_length]
            seqlen = len(words)
            words = words + [0] * max(0, max_sequence_length - seqlen)
            x.append(words)
            tags = tags[:max_sequence_length]
            tags = tags + [0] * max(0, max_sequence_length - len(tags))
            y.append([1, 0])
            input_tags.append(tags)

    with codecs.open(data_path_neg, 'r') as f2:
        for line in f2.readlines():
            line = line.strip()
            words, tags = line.split("#")
            words = words.strip().split()
            tags = tags.strip().split()
            words = words[:max_sequence_length]
            seqlen = len(words)
            words = words + [0] * max(0, max_sequence_length - seqlen)
            x.append(words)
            tags = tags[:max_sequence_length]
            tags = tags + [0] * max(0, max_sequence_length - len(tags))
            y.append([0, 1])
            input_tags.append(tags)

    return np.array(x), np.array(input_tags), np.array(y)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)) / batch_size)
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]