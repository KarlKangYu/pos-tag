import tensorflow as tf
import numpy as np
import data_loader_2 as data_loader
from text_cnn import TextCNN
import sys



def test(pos_file, neg_file, ckpt_path, sequence_length=30, words_vocab_size=50000, tags_vocab_size=51, ensemble=True,
         deps_vocab_size=47, embedding_dim=300, filter_sizes="3,4,5", num_filters=128, tempreture=1):
    # Data Preparation
    # ==================================================

    # Load test data
    print("Loading Test data...")
    x, tags, y = data_loader.read_data(pos_file, neg_file, sequence_length)

    # data_size = len(y)
    # num_batches_per_epoch = (data_size // 256) + 1

    label = np.argmax(y, axis=1)
    neg_y = np.sum(label == 1)
    print("Negative Data Numbers:", neg_y)

    #probabilities = list()

    num_filters = int(num_filters)


    with tf.Graph().as_default():
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=sequence_length,
                num_classes=2,
                vocab_size=words_vocab_size,
                tags_vocab_size=tags_vocab_size,
                deps_vocab_size=deps_vocab_size,
                embedding_size=embedding_dim,
                filter_sizes=list(map(int, filter_sizes.split(","))),
                num_filters=num_filters)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

            sess.run(tf.global_variables_initializer())

            saver.restore(sess=sess, save_path=ckpt_path)

            feed_dict = {
                cnn.input_x: x,
                cnn.input_tags: tags,
                cnn.input_y: y,
                cnn.dropout_keep_prob: 1.0,
                cnn.is_training: False,
                cnn.tempreture: tempreture
            }

            prediction, probability, accuracy = sess.run([cnn.predictions, cnn.probabilities, cnn.accuracy],
                                                         feed_dict=feed_dict)

    count = 0
    pre = np.argmax(prediction, axis=1)
    number = np.sum(pre == 1)
    for i in range(len(pre)):
        if pre[i] == 1 and label[i] == pre[i]:
            count += 1

    recall = count / neg_y
    precision = count / number

    print("Accuracy:", accuracy, "Recall:", recall, "Precision:", precision)


if __name__ == "__main__":
    args = sys.argv
    pos_file = args[1]
    neg_file = args[2]
    tags_vocab_size = args[3]
    ckpt_path = args[4]
    filter_sizes = args[5]
    num_filters = args[6]

    test(pos_file, neg_file, ckpt_path, tags_vocab_size=tags_vocab_size, filter_sizes=filter_sizes, num_filters=num_filters)

