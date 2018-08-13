import tensorflow as tf
import numpy as np
import data_loader
from text_cnn import TextCNN as TextCNN1
from text_cnn_old import TextCNN as TextCNN2
from text_cnn_old_noBN import TextCNN as TextCNN3
import sys
import codecs

def test(pos_file, neg_file, i, ckpt_path, out_dir, sequence_length=30, words_vocab_size=50000, tags_vocab_size=44, ensemble=True,
         deps_vocab_size=47, embedding_dim=300, filter_sizes="3,4,5", num_filters=128, tempreture=20):
    # Data Preparation
    # ==================================================

    # Load test data
    print("Loading Test data...")
    x, tags, deps, heads, y = data_loader.read_data(pos_file, neg_file, sequence_length)

    data_size = len(y)
    num_batches_per_epoch = (data_size // 256) + 1

    label = np.argmax(y, axis=1)
    neg_y = np.sum(label == 1)
    print("Negative Data Numbers:", neg_y)

    #probabilities = list()

    TextCNN = [TextCNN1, TextCNN2, TextCNN3]
    i = int(i)
    num_filters = int(num_filters)
    assert i >= 0 and i <= 2

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN[i](
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


            for i in range(num_batches_per_epoch):
                batch_x = x[i * 256 : (i + 1) * 256]
                batch_tags = tags[i * 256 : (i + 1) * 256]
                batch_deps = deps[i * 256 : (i + 1) * 256]
                batch_heads = heads[i * 256 : (i + 1) * 256]
                batch_y = y[i * 256 : (i + 1) * 256]

                feed_dict = {
                    cnn.input_x: batch_x,
                    cnn.input_tags: batch_tags,
                    cnn.input_deps: batch_deps,
                    cnn.input_head: batch_heads,
                    cnn.input_y: batch_y,
                    cnn.dropout_keep_prob: 1.0,
                    cnn.is_training: False,
                    cnn.tempreture: tempreture
                }

                prediction, probability, accuracy = sess.run([cnn.predictions, cnn.probabilities, cnn.accuracy],
                                                             feed_dict=feed_dict)

                if i == 0:
                    probabilities = probability
                else:
                    probabilities = np.concatenate((probabilities, probability), axis=0)


            # count = 0
            # for i in range(len(prediction)):
            #     if prediction[i] == 1 and prediction[i] == label[i]:
            #         count += 1
            # recall = count / neg_y
            # precision = count / np.sum(prediction == 1)
            # print("Accuracy: {}, Recall:{}, Precision:{}".format(accuracy, recall, precision))
            # print("\n")

            print("probabilities length:", len(probabilities))
            with codecs.open(out_dir, 'w', encoding="utf-8") as f:
                for prob in probabilities:
                    pos, neg = prob
                    pos = str(pos)
                    neg = str(neg)
                    f.write(pos + ',' + neg + '#')
                f.write("\n")




            # for i in range(ckpts_num):
            #     ckpt_path = ckpts_path[i]
            #     saver.restore(sess=sess, save_path=ckpt_path)
            #     print("*" * 20 + "\nLoading The {} Model from {}:\n".format(i+1, ckpt_path))
            #
            #     feed_dict = {
            #         cnn.input_x: x,
            #         cnn.input_tags: tags,
            #         cnn.input_deps: deps,
            #         cnn.input_head: heads,
            #         cnn.input_y: y,
            #         cnn.dropout_keep_prob: 1.0,
            #         cnn.is_training: False
            #     }
            #
            #     prediction, probability, accuracy = sess.run([cnn.predictions, cnn.probabilities, cnn.accuracy], feed_dict=feed_dict)
            #     probabilities.append(probability)
            #
            #     count = 0
            #     for i in range(len(prediction)):
            #         if prediction[i] == 1 and prediction[i] == label[i]:
            #             count += 1
            #     recall = count / neg_y
            #     precision = count / np.sum(prediction == 1)
            #     print("Accuracy: {}, Recall:{}, Precision:{}".format(accuracy, recall, precision))
            #     print("\n")

            # if ensemble:
            #     probabilities = np.array(probabilities)
            #     probability = np.mean(probabilities, axis=0)
            #     assert len(probability) == len(probabilities[0])
            #     pre = np.argmax(probability, 1)
            #     accuracy = np.sum(pre == label) / len(label)
            #     count = 0
            #     for i in range(len(pre)):
            #         if pre[i] == 1 and pre[i] == label[i]:
            #             count += 1
            #     recall = count / neg_y
            #     precision = count / np.sum(pre == 1)
            #
            #     print("*" * 20 + "\nEnsemble Model:\n")
            #     print("Accuracy:", accuracy, "Recall:", recall, "Precision:", precision)
            #     print("\n" + "*" * 20)


if __name__ == "__main__":
    args = sys.argv
    pos_file = args[1]
    neg_file = args[2]
    i = args[3]
    ckpt_path = args[4]
    out_dir = args[5]
    filter_sizes = args[6]
    num_filters = args[7]

    test(pos_file, neg_file, i, ckpt_path, out_dir, filter_sizes=filter_sizes, num_filters=num_filters)

