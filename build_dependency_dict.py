import spacy
import codecs
import sys

def build_dict(data_in, vocab_out):
    dep2ids = {"_PAD": 0, "<unk>": 1}
    i = 2
    with codecs.open(data_in, 'r') as f1:
        nlp = spacy.load("en")
        for line in f1.readlines():
            line = line.strip()
            doc = nlp(line)
            for token in doc:
                dep = token.dep_
                if dep in dep2ids:
                    continue
                else:
                    dep2ids[dep] = i
                    i += 1

    with codecs.open(vocab_out, 'w') as f2:
        for dep, id in dep2ids.items():
            f2.write(dep + "##" + str(id) + '\n')



if __name__ == "__main__":
    args = sys.argv
    data_in = args[1]
    vocab_out = args[2]
    build_dict(data_in, vocab_out)





