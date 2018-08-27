import codecs
import sys

def tok2id(data, vocab_words, vocab_tags, out_data):
    words2ids = {}
    tags2ids = {}
    j = 0
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

    with codecs.open(data, 'r') as f3:
        with codecs.open(out_data, 'w') as f4:
            for line in f3.readlines():
                line = line.strip()

                try:
                    words_line, tags_line = line.split('#')
                    words_line = words_line.strip().split()
                    for word in words_line:
                        id_word = words2ids.get(word, words2ids["<unk>"])
                        f4.write(id_word + ' ')
                    f4.write('#' + ' ')

                    tags_line = tags_line.strip().split()
                    for tag in tags_line:
                        id_tag = tags2ids.get(tag, tags2ids["<unk>"])
                        f4.write(id_tag + ' ')
                    f4.write("\n")
                except:
                    j += 1

    print(j, "lines are skipped!!!")


if __name__ == "__main__":
    args = sys.argv
    data = args[1]
    vocab_words = args[2]
    vocab_tags = args[3]
    out_data = args[4]
    tok2id(data, vocab_words, vocab_tags, out_data)


