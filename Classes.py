
import math

from Util import getDataFromFile
from Util import makeBigramMap
from Util import makeWords
from Util import countWords
from Util import replaceTestData
from Util import replaceTrainingData
from Util import writeToFile

from Util import MODIFIED
from Util import START
from Util import END
from Util import UNK
from Util import UNDEFINED


class PreProcess:
    """PreProcess a file, pad tags, lower lines, count words, replace (depends on other PreProcess object)
    finally write to a file"""

    def __init__(self, filename, otherPreProcess=None):
        self.filename = filename
        self.actualLines = getDataFromFile(self.filename)
        self.actualTokenMap = countWords(makeWords(self.actualLines))
        self.actualTotalToken = sum(self.actualTokenMap.values())
        self.actualUniqueToken = len(self.actualTokenMap.keys())
        if otherPreProcess is None:
            self.replacedLines = replaceTrainingData(self.actualLines, self.actualTokenMap)
        else:
            self.replacedLines = replaceTestData(self.actualLines, otherPreProcess.replacedTokenMap)
        self.replacedTokenMap = countWords(makeWords(self.replacedLines))
        self.replacedTotalToken = sum(self.replacedTokenMap.values())
        self.replacedUniqueToken = len(self.replacedTokenMap.keys())
        self.modifiedFilename = MODIFIED + self.filename
        writeToFile(self.modifiedFilename, self.replacedLines)


class Unigram:
    """ counting padding and <unk> as a token and map them to a dictionary keys and have value as their
        probability under this model"""

    def __init__(self, pre, smoothing=False):
        self.ungTokenMap = pre.replacedTokenMap
        self.ungTotalToken = pre.replacedTotalToken
        self.ungUniqueToken = pre.replacedUniqueToken
        self.smoothing = smoothing

    def calUnigramWordProbability(self, word):
        """:returns probability of the given word"""

        if self.smoothing:
            top = self.ungTokenMap.get(word, 0) + 1
            bottom = self.ungTotalToken + self.ungUniqueToken
            return top/bottom
        else:
            return self.ungTokenMap.get(word, 0)/self.ungTotalToken

    def calUnigramSentenceProbability(self, sentence, padded=False):
        """:returns the probability of a given sentence"""

        totalProbability = 1
        if padded:
            words = sentence.split()
        else:
            words = (START + " " + sentence + " " + END).lower().split()

        for word in words:
            if self.ungTokenMap.get(word, 0) == 0:
                word = UNK
            totalProbability *= self.calUnigramWordProbability(word)
        return totalProbability

    def calUnigramSentencePerplexity(self, sentence, padded=False):
        """:returns the perplexity of a given sentence"""

        totalProbability = 0.0
        if padded:
            words = sentence.split()
        else:
            words = (START + " " + sentence + " " + END).lower().split()

        for word in words:
            if self.ungTokenMap.get(word, 0) == 0:
                word = UNK
            wordProbability = self.calUnigramWordProbability(word)
            totalProbability += math.log(wordProbability, 2)
        return math.pow(2, -(totalProbability/len(words)))

    def calUnigramSentencePerplexityTest(self, sentence, padded=False):
        """:returns the probability of a given sentence for perplexity of a test corpora"""

        totalProbability = 0.0
        if padded:
            words = sentence.split()
        else:
            words = (START + " " + sentence + " " + END).lower().split()
        for word in words:
            if self.ungTokenMap.get(word, 0) == 0:
                word = UNK
            wordProbability = self.calUnigramWordProbability(word)
            totalProbability += math.log(wordProbability, 2)
        return totalProbability


class Bigram(Unigram):
    """counting padding and unk as a token and map them to a dictionary keys and have value as
    their probability under this model"""

    def __init__(self, pre):
        Unigram.__init__(self, pre)
        self.biTokenMap = makeBigramMap(pre.replacedLines)

    def calBigramWordProbability(self, previousWord, word):
        """:returns probability of the given word"""

        top = self.biTokenMap.get((previousWord, word), 0)
        bottom = self.ungTokenMap.get(previousWord, 0)
        if top == 0 or bottom == 0:
            return 0.0
        else:
            return top/bottom

    def calBigramSentenceProbability(self, sentence, padded=False):
        """:returns probability of the given sentence"""

        totalProbability = 1
        if padded:
            words = sentence.split()
        else:
            words = (START + " " + sentence + " " + END).lower().split()
        j = 1
        for i in range(len(words)-1):
            w1 = words[i]
            w2 = words[j]
            if self.ungTokenMap.get(w1, 0) == 0:
                w1 = UNK
            if self.ungTokenMap.get(w2, 0) == 0:
                w2 = UNK
            j += 1
            wordProbability = self.calBigramWordProbability(w1, w2)
            totalProbability *= wordProbability
        return totalProbability

    def calBigramSentencePerplexity(self, sentence, padded=False):
        """:returns the perplexity of a given sentence"""

        totalProbability = 0.0
        if padded:
            words = sentence.split()
        else:
            words = (START + " " + sentence + " " + END).lower().split()
        j = 1
        for i in range(len(words)-1):
            w1 = words[i]
            w2 = words[j]
            if self.ungTokenMap.get(w1, 0) == 0:
                w1 = UNK
            if self.ungTokenMap.get(w2, 0) == 0:
                w2 = UNK
            j += 1
            wordProbability = self.calBigramWordProbability(w1, w2)
            if wordProbability == 0.0:
                return UNDEFINED
            else:
                wordLog = math.log(wordProbability, 2)
            totalProbability += wordLog
        return math.pow(2, -(totalProbability/len(words)))

    def calBigramSentencePerplexityTest(self, sentence, padded=False):
        """:returns the probability of a given sentence for perplexity of a test corpora"""

        totalProbability = 0.0
        if padded:
            words = sentence.split()
        else:
            words = (START + " " + sentence + " " + END).lower().split()
        j = 1
        for i in range(len(words)-1):
            w1 = words[i]
            w2 = words[j]
            if self.ungTokenMap.get(w1, 0) == 0:
                w1 = UNK
            if self.ungTokenMap.get(w2, 0) == 0:
                w2 = UNK
            j += 1
            wordProbability = self.calBigramWordProbability(w1, w2)
            if wordProbability == 0.0:
                return UNDEFINED
            totalProbability += math.log(wordProbability, 2)
        return totalProbability


class BigramSmoothing(Unigram):
    """counting padding and unk as a token and map them to a dictionary keys and have value as
        their probability under this model"""

    def __init__(self, pre):
        Unigram.__init__(self, pre)
        self.bisTokensMap = makeBigramMap(pre.replacedLines)

    def calBigramSmoothingWordProbability(self, previousWord, word):
        """:returns probability of the given word"""

        top = self.bisTokensMap.get((previousWord, word), 0) + 1
        bottom = self.ungTokenMap.get(previousWord, 0) + self.ungUniqueToken
        return top/bottom

    def calBigramSmoothingSentenceProbability(self, sentence, padded=False):
        """:returns probability of the given sentence"""

        probability = 1
        if padded:
            words = sentence.split()
        else:
            words = (START + " " + sentence + " " + END).lower().split()
        j = 1
        for i in range(len(words)-1):
            w1 = words[i]
            w2 = words[j]
            if self.ungTokenMap.get(w1, 0) == 0:
                w1 = UNK
            if self.ungTokenMap.get(w2, 0) == 0:
                w2 = UNK
            j += 1
            probability *= self.calBigramSmoothingWordProbability(w1, w2)
        return probability

    def calBigramSmoothingSentencePerplexity(self, sentence, padded=False):
        """:returns the perplexity of a given sentence"""

        probability = 0.0
        if padded:
            words = sentence.split()
        else:
            words = (START + " " + sentence + " " + END).lower().split()
        j = 1
        for i in range(len(words)-1):
            w1 = words[i]
            w2 = words[j]
            if self.ungTokenMap.get(w1, 0) == 0:
                w1 = UNK
            if self.ungTokenMap.get(w2, 0) == 0:
                w2 = UNK
            j += 1
            wordProb = self.calBigramSmoothingWordProbability(w1, w2)
            if wordProb == 0.0:
                return UNDEFINED
            probability += math.log(wordProb, 2)
        return math.pow(2, -(probability/len(words)))

    def calBigramSmoothingSentencePerplexityTest(self, sentence, padded=False):
        """:returns the probability of a given sentence for perplexity of a test corpora"""

        probability = 0.0
        if padded:
            words = sentence.split()
        else:
            words = (START + " " + sentence + " " + END).lower().split()
        j = 1
        for i in range(len(words)-1):
            w1 = words[i]
            w2 = words[j]
            if self.ungTokenMap.get(w1, 0) == 0:
                w1 = UNK
            if self.ungTokenMap.get(w2, 0) == 0:
                w2 = UNK
            j += 1
            wordProb = self.calBigramSmoothingWordProbability(w1, w2)
            if wordProb == 0.0:
                return UNDEFINED
            probability += math.log(wordProb, 2)
        return probability

