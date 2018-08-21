import codecs
import sys
import spacy

def pos_tagging(data_in, data_out):
    nlp = spacy.load("en")
    with codecs.open(data_in, "r") as f1:
        with codecs.open(data_out, 'w') as f2:
            for line in f1.readlines():
                line = line.strip()
                f2.write(line + "#")
                doc = nlp(line)
                for token in doc:
                    tag = token.tag_
                    f2.write(tag + " ")
                f2.write("\n")


if __name__ == "__main__":
    args = sys.argv
    data_in = args[1]
    data_out = args[2]
    pos_tagging(data_in, data_out)
