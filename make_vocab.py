import codecs
import sys

def make_vocab(pos_tag, detailed, pos, vocab_tag2, vocab_detailed, vocab_pos):
    vocab1 = {"_PAD":0, "<unk>":1}
    vocab2 = {"_PAD":0, "<unk>":1}
    vocab3 = {"_PAD":0, "<unk>":1}
    i = 2
    j = 0
    with codecs.open(pos_tag, "r") as f1:
        for line in f1.readlines():
            line = line.strip()
            try:
                _, toks = line.split("#")
                toks = toks.strip()
                toks = toks.split()
                for tok in toks:
                    if tok not in vocab1:
                        vocab1[tok] = i
                        i += 1
                    else:
                        continue
                j += 1
            except:
                print("file1 No.", j, "line has #")
                j += 1

    j = 0
    i = 2
    with codecs.open(detailed, "r") as f2:
        for line in f2.readlines():
            line = line.strip()
            try:
                _, toks = line.split("#")
                toks = toks.strip()
                toks = toks.split()
                for tok in toks:
                    if tok not in vocab2:
                        vocab2[tok] = i
                        i += 1
                    else:
                        continue
                j += 1
            except:
                print("file2 No.", j, "line has #")
                j += 1

    j = 0
    i = 2
    with codecs.open(pos, "r") as f3:
        for line in f3.readlines():
            line = line.strip()
            try:
                _, toks = line.split("#")
                toks = toks.strip()
                toks = toks.split()
                for tok in toks:
                    if tok not in vocab3:
                        vocab3[tok] = i
                        i += 1
                    else:
                        continue
                j += 1
            except:
                print("file3 No.", j, "line has #")
                j += 1


    with codecs.open(vocab_tag2, 'w') as f4:
        for key in vocab1:
            f4.write(key + "##" + str(vocab1[key]) + "\n")

    with codecs.open(vocab_detailed, 'w') as f5:
        for key in vocab2:
            f5.write(key + "##" + str(vocab2[key]) + "\n")

    with codecs.open(vocab_pos, 'w') as f6:
        for key in vocab3:
            f6.write(key + "##" + str(vocab3[key]) + "\n")




def make_vocab_2(data_in, vocab_tags, vocab_names):
    dict_tags = {"_PAD":0, "<unk>":1}
    dict_names = {"_PAD":0, "<unk>":1}
    i = 2
    j = 2

    with codecs.open(data_in, 'r') as f1:
        for line in f1.readlines():
            line = line.strip()
            _, pos_tags, name_entities = line.split("\t#\t")
            pos_tags = pos_tags.strip().split()
            name_entities = name_entities.strip().split()
            for tag in pos_tags:
                if tag not in dict_tags:
                    dict_tags[tag] = i
                    i += 1
                else:
                    continue

            for name in name_entities:
                if name == "0":
                    name = "<not name entity>"
                if name not in dict_names:
                    dict_names[name] = j
                    j += 1
                else:
                    continue

    with codecs.open(vocab_tags, 'w') as f2:
        for key in dict_tags:
            f2.write(key + "##" + str(dict_tags[key]) + "\n")

    with codecs.open(vocab_names, 'w') as f3:
        for key in dict_names:
            f3.write(key + "##" + str(dict_names[key]) + "\n")



if __name__ == "__main__":
    args = sys.argv
    data_in, vocab_tags, vocab_names = args[1], args[2], args[3]
    make_vocab_2(data_in, vocab_tags, vocab_names)