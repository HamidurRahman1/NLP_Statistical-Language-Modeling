
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


def rep_occ_once(m_count, line_list):
    new_line_list=[]
    new_line=""
    for line in line_list:
        for w in line.split():
            if m_count[w]==1:
                new_line+=" <unk> "
            else:
                new_line+=w+" "
            new_line_list.append(new_line)
            new_line=""
    return new_line_list


def rep_with_train(brown_test_lines_padded, new_brown_train_tokens):
    new_brown_test_lines = []
    for line in brown_test_lines_padded:
        ws = line.split()
        newLine = ""
        for w in ws:
            if w not in new_brown_train_tokens.keys():
                newLine += " <unk> "
            else:
                newLine += w + " "
        new_brown_test_lines.append(newLine.lstrip().rstrip())
    return new_brown_test_lines


def write_into_file(output_file, text_lines):
    temp = open(output_file, "w+")
    for l in text_lines:
        temp.write(l+"\n")


def simpleRead(file):
    llss = []
    f = open(file)
    ls = f.readlines()
    for line in ls:
        llss.append(line.rstrip().lstrip())
    return llss


brown_train = read("brown-train.txt")
brown_train_tokens_wunk = tokenize(brown_train)
new_brown_train = rep_occ_once(brown_train_tokens_wunk, brown_train)
new_brown_train_tokens = tokenize(new_brown_train)
write_into_file("brown-train-out.txt", new_brown_train)

brown_test_lines_padded = read("brown-test.txt")
brown_test_tokens = tokenize(brown_test_lines_padded)
new_brown_test_lines = rep_with_train(brown_test_lines_padded, new_brown_train_tokens)
write_into_file("brown-test-out.txt", new_brown_test_lines)
new_brown_test_tokens = tokenize(new_brown_test_lines)
print(len(new_brown_test_tokens))

