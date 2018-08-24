import codecs
import sys

def make_vocab(pos_tag, detailed, pos, vocab_tag2, vocab_detailed, vocab_pos):
    vocab1 = {"_PAD":0, "<unk>":1}
    vocab2 = {"_PAD":0, "<unk>":1}
    vocab3 = {"_PAD":0, "<unk>":1}
    i = 2
    with codecs.open(pos_tag, "r") as f1:
        for line in f1.readlines():
            line = line.strip()
            _, toks = line.split("#")
            toks = toks.strip()
            toks = toks.split()
            for tok in toks:
                if tok not in vocab1:
                    vocab1[tok] = i
                    i += 1
                else:
                    continue

    i = 2
    with codecs.open(detailed, "r") as f2:
        for line in f2.readlines():
            line = line.strip()
            _, toks = line.split("#")
            toks = toks.strip()
            toks = toks.split()
            for tok in toks:
                if tok not in vocab2:
                    vocab2[tok] = i
                    i += 1
                else:
                    continue

    i = 2
    with codecs.open(pos, "r") as f3:
        for line in f3.readlines():
            line = line.strip()
            _, toks = line.split("#")
            toks = toks.strip()
            toks = toks.split()
            for tok in toks:
                if tok not in vocab3:
                    vocab3[tok] = i
                    i += 1
                else:
                    continue


    with codecs.open(vocab_tag2, 'w') as f4:
        for key in vocab1:
            f4.write(key + "##" + str(vocab1[key]) + "\n")

    with codecs.open(vocab_detailed, 'w') as f5:
        for key in vocab2:
            f5.write(key + "##" + str(vocab2[key]) + "\n")

    with codecs.open(vocab_pos, 'w') as f6:
        for key in vocab3:
            f6.write(key + "##" + str(vocab3[key]) + "\n")


if __name__ == "__main__":
    args = sys.argv
    pos_tag, detailed, pos, vocab_tag2, vocab_detailed, vocab_pos = args[1], args[2], args[3], args[4], args[5], args[6]
    make_vocab(pos_tag, detailed, pos, vocab_tag2, vocab_detailed, vocab_pos)