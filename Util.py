
import math

BROWN_TRAINING = "brown-train.txt"
BROWN_TEST = "brown-test.txt"
LEARNER_TEST = "learner-test.txt"

MODIFIED = "modified-"

START = "<s>"
END = "</s>"
UNK = "<unk>"

S1 = "He was laughed off the screen ."
S2 = "There was no compulsion behind them ."
S3 = "I look forward to hearing your reply ."

UNDEFINED = "undefined"


def getDataFromFile(filePath):
    """:returns a list of all lowered, and padded lines from the given filePath. """

    file = open(filePath)
    lines = file.readlines()
    stripped = list()
    for line in lines:
        stripped.append(START + " " + line.rstrip().lower() + " " + END)
    file.close()
    return stripped


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

    wordFreq = dict()
    for word in words:
        try:
            wordFreq[word] += 1
        except KeyError:
            wordFreq[word] = 1
    return wordFreq


def replaceTrainingData(lines, trainingMap):
    """given lines and a dictionary, replace all words with
    <unk> if that has occurred only once in the map and returns a new list of newly mapped <unK> lines"""

    replacedLines = list()
    for line in lines:
        newLine = ""
        words = line.split()
        for word in words:
            if trainingMap[word] == 1:
                newLine += UNK + " "
            else:
                newLine += word + " "
        replacedLines.append(newLine.rstrip())
    return replacedLines


def replaceTestData(lines, replacedTrainingMap):
    """givens a dictionary and lines, replace all words with
        <unk> if that did not occur in the map and returns a new list of lines"""

    replacedLines = list()
    for line in lines:
        newLine = ""
        words = line.split()
        for word in words:
            if word not in replacedTrainingMap.keys():
                newLine += UNK + " "
            else:
                newLine += word + " "
        replacedLines.append(newLine.rstrip())
    return replacedLines


def getNonMatching(trainingKeys, testKeys, testMap):
    """given two key sets - training, test and a dictionary, compare and returns how many keys did
        not match in both and if not matched then sum the value from the given dictionary and returns
        a tuple of unique keys and total occurrences"""

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


def getPercentage(top, bottom):
    """given a fraction it returns a percentage of that fraction"""
    return (top/bottom)*100


def returnLogProbability(probability):
    """given a probability it returns a log of that probability"""

    if probability <= 0.0:
        return UNDEFINED
    else:
        return math.log(probability, 2)


def allSentencesPerplexityUnderUnigram(lines, unigram, totalTokens):
    """given a list of lines, a Unigram model object, and total tokens it returns the
        total perplexity of the given lines"""

    totalProbability = 0.0
    for line in lines:
        sentenceProbability = unigram.calUnigramSentencePerplexityTest(line, True)
        if undefinedPerplexity(sentenceProbability):
            return UNDEFINED
        totalProbability += sentenceProbability
    return math.pow(2, -(totalProbability/totalTokens))


def allSentencesPerplexityUnderBigram(lines, bigram, totalTokens):
    """given a list of lines, a Bigram model object, and total tokens it returns the
        total perplexity of the given lines"""

    totalProbability = 0.0
    for line in lines:
        sentenceProbability = bigram.calBigramSentencePerplexityTest(line, True)
        if undefinedPerplexity(sentenceProbability):
            return UNDEFINED
        totalProbability += sentenceProbability
    return math.pow(2, -(totalProbability/totalTokens))


def allSentencesPerplexityUnderBigramSmoothing(lines, bigramSmoothing, totalTokens):
    """given a list of lines, a BigramSmoothing model object, and total tokens it returns the
        total perplexity of the given lines"""

    totalProbability = 0.0
    for line in lines:
        sentenceProbability = bigramSmoothing.calBigramSmoothingSentencePerplexityTest(line, True)
        if undefinedPerplexity(sentenceProbability):
            return UNDEFINED
        totalProbability += sentenceProbability
    return math.pow(2, -(totalProbability/totalTokens))


def undefinedPerplexity(probability):
    """:returns if the given probability is true towards perplexity counting"""

    if probability == UNDEFINED:
        return True
    else:
        return False


def padLines(lines):
    """given original lines this function modify those lines by padding them and return given list"""

    i = 0
    for line in lines:
        line = START + " " + line + " " + END
        lines[i] = line
        i += 1
    return lines


def matched(trainingMap, testMap):
    """given two dictionary training and test, compare and returns how many keys matched in both dictionary"""

    counter = 0
    for k in trainingMap.keys():
        if k in testMap.keys():
            counter += testMap[k]
    return counter


def lowerAll(lines):
    """given a list of lines it's lowers all the lines and returns a new list containing those"""

    lowered = list()
    for line in lines:
        lowered.append(line.lower())
    return lowered


def testFileProbabilityOfSentences(lines, modelObj, models):
    totalProbability = 1
    if isinstance(modelObj, models[0]):
        for line in lines:
            totalProbability *= modelObj.calUniSentProb(line, True)
        return totalProbability
    elif isinstance(modelObj, models[1]):
        for line in lines:
            totalProbability *= modelObj.calBiSentProb(line, True)
        return totalProbability
    elif isinstance(modelObj, models[2]):
        for line in lines:
            totalProbability *= modelObj.calBisSentProb(line, True)
        return totalProbability

