
def tokenize(all_texts_lines):
    word_map = {}
    for l in all_texts_lines:
        words = l.split()
        for w in words:
            if w in word_map.keys():
                word_map[w] += 1
            else:
                word_map[w] = 1
    return word_map


def read(file):
    f = open(file)
    all = f.readlines()
    newFile = []
    for l in all:
        newFile.append("<s> " + l.rstrip().lstrip().lower() + " </s>")
    return newFile


def modifiy(oldMap, taggedLines):
    lst = []
    newLine = ""
    for line in taggedLines:
        ws = line.split()
        for w in ws:
            if oldMap[w] == 1:
                newLine += " <unk> "
            else:
                newLine += w + " "
        lst.append(newLine)
        newLine = ""
    return lst


updatedLines = read("brown-train.txt")
upm = tokenize(updatedLines)
print("Total tokens", sum(upm.values()))