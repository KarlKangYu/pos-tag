import tensorflow as tf
import numpy as np
import data_loader
from text_cnn import TextCNN
import sys

def test(pos_file, neg_file, ckpt1, ckpt2, sequence_length=30, words_vocab_size=50000, tags_vocab_size=44, ensemble=True,
         deps_vocab_size=47, embedding_dim=300, filter_sizes="3,4,5", num_filters=128):
    # Data Preparation
    # ==================================================

    # Load test data
    print("Loading Test data...")
    x, tags, deps, heads, y = data_loader.read_data(pos_file, neg_file, sequence_length)

    label = np.argmax(y, axis=1)
    neg_y = np.sum(label == 1)
    print("Negative Data Numbers:", neg_y)

    probabilities = list()

    with tf.Graph().as_default():
        sess = tf.Session()
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

            saver.restore(sess=sess, save_path=ckpt1)

            print("*" * 20 + "\nFirst Model:\n")

            feed_dict = {
                cnn.input_x: x,
                cnn.input_tags: tags,
                cnn.input_deps: deps,
                cnn.input_head: heads,
                cnn.input_y: y,
                cnn.dropout_keep_prob: 1.0
            }

            predictions1, probability1, accuracy1 = sess.run([cnn.predictions, cnn.probabilities, cnn.accuracy], feed_dict=feed_dict)
            probabilities.append(probability1)

            count=0
            for i in range(len(predictions1)):
                if predictions1[i] == 1 and predictions1[i] == label[i]:
                    count += 1
            recall = count/neg_y

            neg_pre1 = np.sum(predictions1 == 1)
            print("In prediction, Negative number:", neg_pre1)
            precision = count/neg_pre1

            print("Accuracy:", accuracy1, "Recall:", recall, "Precision:", precision, "\n" + "*" * 20)

            ######################################################################################################

            saver.restore(sess=sess, save_path=ckpt2)

            print("*" * 20 + "\nSecond Model:\n")

            feed_dict = {
                cnn.input_x: x,
                cnn.input_tags: tags,
                cnn.input_deps: deps,
                cnn.input_head: heads,
                cnn.input_y: y,
                cnn.dropout_keep_prob: 1.0
            }

            predictions2, probability2, accuracy2 = sess.run([cnn.predictions, cnn.probabilities, cnn.accuracy],
                                                             feed_dict=feed_dict)
            probabilities.append(probability2)

            count = 0
            for i in range(len(predictions2)):
                if predictions2[i] == 1 and predictions2[i] == label[i]:
                    count += 1
            recall = count / neg_y

            neg_pre2 = np.sum(predictions2 == 1)
            print("In prediction, Negative number:", neg_pre2)
            precision = count / neg_pre2

            print("Accuracy:", accuracy2, "Recall:", recall, "Precision:", precision, "\n" + "*" * 20)

            if ensemble:
                probabilities = np.array(probabilities)
                probability = np.mean(probabilities, axis=0)
                assert len(probability) == len(probabilities[0])
                pre = np.argmax(probability, 1)
                accuracy = np.sum(pre == label) / len(label)
                count = 0
                for i in range(len(pre)):
                    if pre[i] == 1 and pre[i] == label[i]:
                        count += 1
                recall = count / neg_y
                precision = count / np.sum(pre == 1)

                print("*" * 20 + "\nEnsemble Model:\n")
                print("Accuracy:", accuracy, "Recall:", recall, "Precision:", precision)
                print("\n" + "*" * 20)

if __name__ == "__main__":
    args = sys.argv
    pos_file = args[1]
    neg_file = args[2]
    ckpt1 = args[3]
    ckpt2 = args[4]
    test(pos_file, neg_file, ckpt1, ckpt2)

