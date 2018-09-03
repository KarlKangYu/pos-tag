import codecs
import numpy as np
import sys

def read_prob(prob_path):
    label = [0] * 4158 + [1] * 4158
    label = np.array(label)
    for i in range(1, 8):
        path = prob_path + str(i)
        with codecs.open(path, 'r') as f:
            prob = f.readline()
            prob = prob.strip().strip('#')
            prob = prob.split('#')
            b = []  # b里包括一个文件（一个checkpoint）跑出的所有结果
            for pro in prob:
                pos, neg = pro.split(',')
                pos = float(pos)
                neg = float(neg)
                a = [pos, neg]  # a里包括一个样本跑出的结果
                b.append(a)
        pre = np.argmax(b, axis=1)
        assert len(pre) == len(label)
        count = 0
        for j in range(len(pre)):
            if pre[j] == label[j]:
                count += 1
        acc = count / len(pre)

        count = 0
        for j in range(len(pre)):
            if pre[j] == label[j] and pre[j] == 1:
                count += 1
        recall = count / 4158
        precision = count / np.sum(pre == 1)
        print("No.", i, "Model's Accuracy:", acc, "Recall:", recall, "Precision:", precision)


if __name__ == "__main__":
    args = sys.argv
    prob_path = args[1]
    read_prob(prob_path)

            