
import math

from Util import getDataFromFile
from Util import makeBigramMap
from Util import makeWords
from Util import countWords
from Util import replaceAndPaddTest
from Util import replaceAndPaddTraining
from Util import writeToFile

from Util import MODIFIED
from Util import START
from Util import END
from Util import UNK


class PreProcess:
    """PreProcess a file, pad tags, lower lines, count words, replace (depends on other PreProcess object)
    finally write to a file"""

    def __init__(self, filename, other=None):
        self.filename = filename
        self.actualLines = getDataFromFile(self.filename)
        self.actualTokenMap = countWords(makeWords(self.actualLines))
        self.actualTotalToken = sum(self.actualTokenMap.values())
        self.actualUniqueToken = len(self.actualTokenMap.keys())
        if other is None:
            self.replacedLines = replaceAndPaddTraining(self.actualLines, self.actualTokenMap)
        else:
            self.replacedLines = replaceAndPaddTest(self.actualLines, other.replacedTokenMap)
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

    def calUngWordProb(self, word):
        """:returns probability of the given word"""

        if self.smoothing:
            top = self.ungTokenMap.get(word, 0) + 1
            bottom = self.ungTotalToken + self.ungUniqueToken
            return top/bottom
        else:
            return self.ungTokenMap.get(word, 0)/self.ungTotalToken

    def calUniSentProb(self, sentence, padded=False):
        """:returns the probability of a given sentence"""

        totalProb = 1
        if padded:
            words = sentence.split()
        else:
            words = (START + " " + sentence + " " + END).lower().split()

        for word in words:
            if self.ungTokenMap.get(word, 0) == 0:
                word = UNK
            wordProb = self.calUngWordProb(word)
            totalProb *= wordProb
        return totalProb

    def calUniSentPerplexity(self, sentence, padded=False):
        """:returns the probability of a given sentence"""

        totalProb = 1
        if padded:
            words = sentence.split()
        else:
            words = (START + " " + sentence + " " + END).lower().split()

        for word in words:
            if self.ungTokenMap.get(word, 0) == 0:
                word = UNK
            wordProb = self.calUngWordProb(word)
            totalProb += wordProb
        return totalProb


class Bigram(Unigram):
    """counting padding and unk as a token and map them to a dictionary keys and have value as
    their probability under this model"""

    def __init__(self, pre):
        Unigram.__init__(self, pre)
        self.biTokenMap = makeBigramMap(pre.replacedLines)

    def calBiWordProb(self, previousWord, word):
        top = self.biTokenMap.get((previousWord, word), 0)
        bottom = self.ungTokenMap.get(previousWord, 0)
        if top == 0 or bottom == 0:
            return 0.0
        else:
            return top/bottom

    def calBiSentProb(self, sentence, padded=False):

        totalProb = 1
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
            wordProb = self.calBiWordProb(w1, w2)
            totalProb *= wordProb
        return totalProb

    def calBiSentPerplexity(self, sentence, padded=False):

        totalProb = 1
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
            wordProb = self.calBiWordProb(w1, w2)
            totalProb += wordProb
        return totalProb


class BigramSmoothing(Unigram):
    def __init__(self, pre):
        Unigram.__init__(self, pre)
        self.bisTokensMap = makeBigramMap(pre.replacedLines)

    def calBisWordProb(self, previousWord, word):
        top = self.bisTokensMap.get((previousWord, word), 0) + 1
        bottom = self.ungTokenMap.get(previousWord, 0) + self.ungUniqueToken
        return top/bottom

    def calBisSentProb(self, sentence, padded=False):
        prob = 1
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
            prob *= self.calBisWordProb(w1, w2)
        return prob

    def calBisSentPerplexity(self, sentence, padded=False):
        prob = 1
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
            prob *= self.calBisWordProb(w1, w2)
        return prob

