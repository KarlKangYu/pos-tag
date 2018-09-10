import codecs
import sys

def tok2id(data, vocab_words, vocab_tags, vocab_names, out_data):
    words2ids = {}
    tags2ids = {}
    names2ids = {}

    with codecs.open(vocab_words, 'r') as f1:
        for line in f1.readlines():
            line = line.strip()
            word, id = line.split("##")
            words2ids[word] = id

    with codecs.open(vocab_tags, 'r') as f2:
        for line in f2.readlines():
            line = line.strip()
            tag, id = line.split("##")
            tags2ids[tag] = id

    with codecs.open(vocab_names, 'r') as f5:
        for line in f5.readlines():
            line = line.strip()
            name, id = line.split("##")
            if name == "<not name entity>":
                name = "0"
            names2ids[name] = id

    with codecs.open(data, 'r', encoding="utf-8") as f3:
        with codecs.open(out_data, 'w', encoding="utf-8") as f4:
            for line in f3.readlines():
                line = line.strip()

                words_line, tags_line, names_line = line.split('\t#\t')
                words_line = words_line.strip().split()
                for word in words_line:
                    id_word = words2ids.get(word, words2ids["<unk>"])
                    f4.write(id_word + ' ')
                f4.write('\t#\t' + ' ')

                tags_line = tags_line.strip().split()
                for tag in tags_line:
                    id_tag = tags2ids.get(tag, tags2ids["<unk>"])
                    f4.write(id_tag + ' ')
                f4.write("\t#\t" + ' ')

                names_line = names_line.strip().split()
                for name in names_line:
                    id_name = names2ids.get(name, names2ids["<unk>"])
                    f4.write(id_name + ' ')
                f4.write("\n")


if __name__ == "__main__":
    args = sys.argv
    data = args[1]
    vocab_words = args[2]
    vocab_tags = args[3]
    vocab_names = args[4]
    out_data = args[5]
    tok2id(data, vocab_words, vocab_tags, vocab_names, out_data)


