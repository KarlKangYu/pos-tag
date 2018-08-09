import codecs
import numpy as np
import data_loader
import sys

def ensemble(pos_file, neg_file, i, dir, sequence_length=30):
    probabilities = list()
    i = int(i)
    for i in range(i):
        path = dir + str(i+1)
        with codecs.open(path, 'r', encoding="utf-8") as f:
            prob = f.readline()
            prob = prob.strip()
            prob = prob.split('#')
            b = []
            for pro in prob:
                pos, neg = pro.split(',')
                pos = float(pos)
                neg = float(neg)
                a = [pos, neg]
                b.append(a)
        probabilities.append(b)

    x, tags, deps, heads, y = data_loader.read_data(pos_file, neg_file, sequence_length)
    label = np.argmax(y, axis=1)
    neg_y = np.sum(label == 1)

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
    i = args[3]
    dir = args[4]
    ensemble(pos_file, neg_file, i, dir)


