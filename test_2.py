import tensorflow as tf
import numpy as np
import data_loader_2 as data_loader
from text_cnn_2layers import TextCNN
import sys
import codecs


def test(pos_file, neg_file, ckpt_path, out, sequence_length=30, words_vocab_size=50000, tags_vocab_size=51,
         name_vocab_size=20, embedding_dim=300, filter_sizes="3,5,7", filter_sizes2="3,5,7,9,11,13", num_filters=256, tempreture=1):
    # Data Preparation
    # ==================================================

    # Load test data
    print("Loading Test data...")
    x, tags, names, y = data_loader.read_data(pos_file, neg_file, sequence_length)

    # data_size = len(y)
    # num_batches_per_epoch = (data_size // 256) + 1

    label = np.argmax(y, axis=1)
    neg_y = np.sum(label == 1)
    print("Negative Data Numbers:", neg_y)

    #probabilities = list()

    # num_filters = int(num_filters)


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
                name_vocab_size=name_vocab_size,
                embedding_size=embedding_dim,
                filter_sizes=list(map(int, filter_sizes.split(","))),
                filter_sizes2=list(map(int, filter_sizes2.split(","))),
                num_filters=num_filters)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

            sess.run(tf.global_variables_initializer())

            saver.restore(sess=sess, save_path=ckpt_path)

            feed_dict = {
                cnn.input_x: x,
                cnn.input_tags: tags,
                cnn.input_name_entity: names,
                cnn.input_y: y,
                cnn.dropout_keep_prob_1: 1.0,
                cnn.dropout_keep_prob_2: 1.0,
                cnn.is_training: False,
                cnn.tempreture: tempreture
            }

            prediction, probability, accuracy = sess.run([cnn.predictions, cnn.probabilities, cnn.accuracy],
                                                         feed_dict=feed_dict)

    count = 0
    pre = prediction
    number = np.sum(pre == 1)
    for i in range(len(pre)):
        if pre[i] == 1 and label[i] == pre[i]:
            count += 1

    recall = count / neg_y
    precision = count / number

    print("Accuracy:", accuracy, "Recall:", recall, "Precision:", precision)

    # index = []
    # for i in range(len(pre)):
    #     if pre[i] != label[i]:
    #         if i < 4158:
    #             index.append([i, 4158+i])
    #         else:
    #             index.append([4158-i, i])
    #
    # ff = codecs.open(pos_data, 'r')
    # a1 = ff.readlines()
    # ff.close()
    # ff = codecs.open(neg_data, 'r')
    # a2 = ff.readlines()
    # ff.close()
    # a1 = a1 + a2
    # del a2
    # assert len(a1) == len(pre)
    # with codecs.open(out, 'w') as ff:
    #     for ind in index:
    #         ind1, ind2 = ind
    #         posline, negline = a1[ind1], a1[ind2]
    #         ff.write(posline.strip() + "#" * 10 + negline.strip() + "\n")
    #

    print("Probability Shape:", probability.shape)
    with codecs.open(out, 'w') as f:
        for prob in probability:
            pos, neg = prob
            pos, neg = str(pos), str(neg)
            f.write(pos + '##' + neg + "\n")



if __name__ == "__main__":
    args = sys.argv
    pos_file = args[1]
    neg_file = args[2]
    ckpt_path = args[3]

    out = args[4]

    test(pos_file, neg_file, ckpt_path, out=out)

