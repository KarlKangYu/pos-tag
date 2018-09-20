import numpy as np
# from sklearn.metrics import roc_auc_score, roc_curve
import codecs
import data_loader_2 as data_loader
import sys


# def auc(pos_file, neg_file, test_prob_path, sequence_length=30):
#     x, tags, deps, heads, y = data_loader.read_data(pos_file, neg_file, sequence_length)
#     y_true = np.argmax(y, axis=1)  #pos: 0, neg: 1
#     score = list()
#     with codecs.open(test_prob_path, 'r') as f:
#         for line in f.readlines():
#             line = line.strip()
#             _, neg = line.split(",")
#             neg = float(neg)
#             score.append(neg)
#
#     y_score = np.array(score)
#
#     assert len(y_score) == len(y_true)
#     print("Shape:", y_score.shape)
#     auc = roc_auc_score(y_true, y_score)
#     print("AUC:", auc)


def test_recall(pos_file, neg_file, test_prob_path, threshold, sequence_length=30):
    threshold = float(threshold)
    x, tags, names, y = data_loader.read_data(pos_file, neg_file, sequence_length)
    y_true = np.argmax(y, axis=1) #pos=0, neg=1
    num_data = len(y_true)
    print("Data Number: ", num_data)
    num_pos = np.sum(y_true == 0)
    num_neg = num_data - num_pos
    print("Positive Number:", num_pos, "Negative Number:", num_neg)
    predict = list()
    with codecs.open(test_prob_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            pos, neg = line.split("##")
            pos, neg = float(pos), float(neg)
            if pos > threshold:
                predict.append(0)
            else:
                predict.append(1)
    y_pre = np.array(predict)

    assert len(y_true) == len(y_pre)
    accuracy = np.sum(y_true == y_pre) / len(y_true)
    count = 0
    for i in range(len(y_pre)):
        if y_pre[i] == 1 and y_pre[i] == y_true[i]:
            count += 1

    recall = count / np.sum(y_true==1)
    precision = count / np.sum(y_pre==1)

    tp = 0
    for i in range(len(y_pre)):
        if y_pre[i] == 0 and y_pre[i] == y_true[i]:
            tp += 1

    tn = count
    fp = num_neg - tn
    fn = num_pos - tp

    print("Accuracy:", accuracy, "Recall:", recall, "Precision:", precision)
    print("TP:", tp, "FP:", fp, "\n", "FN:", fn, "TN:", tn )


if __name__ == "__main__":
    args = sys.argv
    pos_file = args[1]
    neg_file = args[2]
    test_prob_path = args[3]
    threshold = args[4]
    test_recall(pos_file, neg_file, test_prob_path, threshold)

