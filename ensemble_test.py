import codecs
import numpy as np
import data_loader_test as data_loader
import sys

def ensemble(pos_file, neg_file, i, dir, soft_target_path, sequence_length=30):
    probabilities = list()
    i = int(i)
    for i in range(i):
        path = dir + str(i+1)
        with codecs.open(path, 'r', encoding="utf-8") as f:
            prob = f.readline()
            prob = prob.strip().strip('#')
            prob = prob.split('#')
            b = []        #b里包括一个文件（一个checkpoint）跑出的所有结果
            for pro in prob:
                pos, neg = pro.split(',')
                pos = float(pos)
                neg = float(neg)
                a = [pos, neg]     #a里包括一个样本跑出的结果
                b.append(a)
        probabilities.append(b)

    x, tags, deps, heads, y = data_loader.read_data(pos_file, neg_file, sequence_length)
    label = np.argmax(y, axis=1)
    neg_y = np.sum(label == 1)

    probabilities = np.array(probabilities)
    print(probabilities.shape)
    probability = np.mean(probabilities, axis=0)
    assert len(probability) == len(probabilities[0])
    pre = np.argmax(probability, axis=1)
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

    print("-" * 20 + "\nWriting soft-target!\n" + "-" * 20)
    assert len(probability) == len(y) and len(pre) == len(label)
    print("Data Numbers:", len(probability))
    with codecs.open(soft_target_path, "w", encoding="utf-8") as ff:
        for i in range(len(pre)):
            if pre[i] == label[i]:
                soft_target = 0.1 * y[i] + 0.9 * probability[i]
            else:
                soft_target = 0.4 * y[i] + 0.6 * probability[i]

            pos, neg = soft_target
            pos, neg = str(pos), str(neg)
            ff.write(pos + "," + neg + "\n")


if __name__ == "__main__":
    args = sys.argv
    pos_file = args[1]
    neg_file = args[2]
    i = args[3]
    dir = args[4]
    soft_target_path = args[5]
    ensemble(pos_file, neg_file, i, dir, soft_target_path)


