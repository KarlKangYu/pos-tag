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
                    if token.pos_ in ['VERB', 'ADJ', 'ADV', 'NOUN', 'PROPN'] and token.tag_ not in ['AFX', 'EX', 'HVS', 'PRP$', 'WDT', 'WP', 'WP$', 'WRB']:
                        if token.text in ['am', 'is', 'are', 'was', 'were']:
                            tag = token.text
                        else:
                            tag = token.tag_
                    else:
                        tag = token.text

                    f2.write(tag + " ")
                f2.write("\n")

def pos(data_in, data_out):
    nlp = spacy.load("en")
    with codecs.open(data_in, "r") as f1:
        with codecs.open(data_out, 'w') as f2:
            for line in f1.readlines():
                line = line.strip()
                f2.write(line + "#")
                doc = nlp(line)
                for token in doc:
                    pos = token.pos_
                    f2.write(pos + ' ')
                f2.write("\n")

def tagging(data_in, data_out):
    nlp = spacy.load("en")
    with codecs.open(data_in, "r") as f1:
        with codecs.open(data_out, "w") as f2:
            for line in f1.readlines():
                line = line.strip()
                f2.write(line + "#")
                doc = nlp(line)
                for token in doc:
                    tag = token.tag_
                    f2.write(tag + ' ')
                f2.write("\n")

def name_entity(data_in, data_out):
    nlp = spacy.load("en")
    with codecs.open(data_in, "r") as f1:
        with codecs.open(data_out, "w") as f2:
            for line in f1.readlines():
                line = line.strip()
                f2.write(line + "\t#\t")
                doc = nlp(line)
                for token in doc:
                    tag = token.tag_
                    f2.write(tag + ' ')
                f2.write("\t#\t")
                l = len(doc)
                i = 0
                for ent in doc.ents:
                    while i < l:
                        if ent.text == doc[i].text:
                            f2.write(ent.label_ + ' ')
                            i += 1
                            break
                        else:
                            f2.write('0' + ' ')
                            i += 1
                if i != l:
                    f2.write("0 " * (l - i))
                f2.write("\n")




if __name__ == "__main__":
    args = sys.argv
    data_in = args[1]
    data_out = args[2]
    i = args[3]
    i = int(i)
    if i == 1:
        pos_tagging(data_in, data_out)
    elif i == 2:
        pos(data_in, data_out)
    elif i == 3:
        tagging(data_in, data_out)
    elif i == 4:
        name_entity(data_in, data_out)
