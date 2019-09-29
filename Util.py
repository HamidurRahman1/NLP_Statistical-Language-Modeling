
BROWN_TRAINING = "brown-train.txt"
BROWN_TEST = "brown-test.txt"
LEARNER_TEST = "learner-test.txt"

# modified file indicator, will prepend with filename above
MODIFIED = "m-"

START = "<s>"
END = "</s>"
UNK = "<unk>"


def getDataFromFile(filePath):
    """:returns a list of all lines from the given file_path"""

    file = open(filePath)
    lines = file.readlines()
    stripped = list()
    for line in lines:
        stripped.append(START + " " + line.rstrip().lower() + " " + END)
    file.close()
    return stripped


def padLines(lines):
    i = 0
    for line in lines:
        line = START + " " + line + " " + END
        lines[i] = line
        i += 1
    return lines


def lowerAll(lines):
    """given a list of lines it's lowers all the lines and returns a new list containing those"""

    lowered = list()
    for line in lines:
        lowered.append(line.lower())
    return lowered


def makeWords(lines):
    """:returns all words(tokens) from the given list as words list"""

    words = list()
    for line in lines:
        splited = line.split()
        filtered = list(filter(None, splited))
        words.extend(filtered)
    return words


def writeToFile(fileName, lines):
    """given lines list and filename, all the lines will be written to the file"""

    file = open(fileName, "w+")
    for line in lines:
        file.write(line+"\n")
    file.close()


def countWords(words):
    """:returns a dictionary of each word occurrences from given a list of words"""

    word_freq = dict()
    for word in words:
        try:
            word_freq[word] += 1
        except KeyError:
            word_freq[word] = 1
    return word_freq


def replaceAndPaddTraining(lines, initialMap):
    """given lines and a dictionary, replace all words with
    <unk> if that has occurred only once in the map and returns a new list of lines"""

    padded = list()
    for line in lines:
        newLine = ""
        words = line.split()
        for word in words:
            if initialMap[word] == 1:
                newLine += UNK + " "
            else:
                newLine += word + " "
        padded.append(newLine.rstrip().lstrip())
    return padded


def replaceAndPaddTest(lines, trainingWordMap):
    """givens a dictionary and lines, replace all words with
        <unk> if that did not occur in the map and returns a new list of lines"""

    padded = list()
    for line in lines:
        newLine = ""
        words = line.split()
        for word in words:
            if word not in trainingWordMap.keys():
                newLine += UNK + " "
            else:
                newLine += word + " "
        padded.append(newLine.lstrip().rstrip())
    return padded


def matched(trainingMap, testMap):
    """given two dictionary training and test, compare and returns how many keys matched in both dictionary"""

    counter = 0
    for k in trainingMap.keys():
        if k in testMap.keys():
            counter += testMap[k]
    return counter


def getNonMatching(trainingKeys, testKeys, testMap):
    """given two key sets training, test and a dictionary, compare and returns how many keys did not match in both
    and if not matched then sum the value from the given dictionary and returns a tuple of unique keys and total occurrences"""

    types, tokens = 0, 0
    for k in testKeys:
        if k not in trainingKeys:
            types += 1
            tokens += testMap[k]
    return types, tokens


def makeBigramMap(lines):
    """given a list of lines it makes a Bigram dictionary and returns it"""

    biMap = dict()
    for line in lines:
        end = False
        words = line.split()
        previousWord = words.pop(0)
        for word in words:
            if not end:
                try:
                    biMap[(previousWord, word)] += 1
                except KeyError:
                    biMap[(previousWord, word)] = 1
            previousWord = word
            if previousWord == END:
                end = True
    return biMap


def getPercentage(up, bottom):
    return (up/bottom)*100

