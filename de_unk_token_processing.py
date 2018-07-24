import codecs
import sys

def de_processing(pos_in, neg_in, pos_out, neg_out):
    f1 = codecs.open(pos_in, 'r')
    pos_lines = f1.readlines()
    f2 = codecs.open(neg_in, 'r')
    neg_lines = f2.readlines()
    l = len(pos_lines)
    f3 = codecs.open(pos_out, 'w')
    f4 = codecs.open(neg_out, 'w')
    for i in range(l):
        pos_line = pos_lines[i].strip()
        neg_line = neg_lines[i].strip()
        if pos_line == neg_line:
            continue

        pos_words = pos_line.split()
        FLAG1 = True
        while FLAG1:
            try:
                start = pos_words.index("<u>")
                end = pos_words.index("</u>")
                pos_words = pos_words[:start] + [''.join(pos_words[start+1 : end])] + pos_words[end+1:]
            except:
                FLAG1 = False
        pos_write_line = ' '.join(pos_words)
        f3.write(pos_write_line + "\n")

        neg_words = neg_line.split()
        FLAG2 = True
        while FLAG2:
            try:
                start = neg_words.index("<u>")
                end = neg_words.index("</u>")
                neg_words = neg_words[:start] + [''.join(neg_words[start+1 : end])] + neg_words[end+1:]
            except:
                FLAG2 = False
        neg_write_line = ' '.join(neg_words)
        f4.write(neg_write_line + "\n")

if __name__ == "__main__":
    args = sys.argv
    pos_in = args[1]
    neg_in = args[2]
    pos_out = args[3]
    neg_out = args[4]
    de_processing(pos_in, neg_in, pos_out, neg_out)



