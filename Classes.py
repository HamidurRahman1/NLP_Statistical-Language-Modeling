
from Util import getDataFromFile
from Util import makeBigramMap
from Util import makeWords
from Util import countWords
from Util import lowerAll
from Util import replaceAndPaddTest
from Util import replaceAndPaddTraining
from Util import writeToFile

from Util import M
from Util import START
from Util import END


class PreProcess:
    """PreProcess a file, pad tags, lower lines, count words, replace (depends on other PreProcess object finally write
    to a file"""

    def __init__(self, filename, other = None):
        self.filename = filename
        self.initialLines = getDataFromFile(self.filename)
        self.initialTokensMap = countWords(makeWords(self.initialLines))
        self.initialTotalTokens = sum(self.initialTokensMap.values())
        self.initialUniqueTokens = len(self.initialTokensMap.keys())
        self.loweredActualLines = lowerAll(self.initialLines)
        self.loweredTokensMap = countWords(makeWords(self.loweredActualLines))
        self.loweredTotalTokens = sum(self.loweredTokensMap.values())
        self.loweredUniqueTokens = len(self.loweredTokensMap.keys())
        if other is None:
            self.replacedPaddedLines = replaceAndPaddTraining(self.loweredActualLines, self.loweredTokensMap)
        else:
            self.replacedPaddedLines = replaceAndPaddTest(other.replacedTokensMap, self.loweredActualLines)
        self.replacedTokensMap = countWords(makeWords(self.replacedPaddedLines))
        self.replacedTotalTokens = sum(self.replacedTokensMap.values())
        self.replacedUniqueTokens = len(self.replacedTokensMap.keys())
        self.modifiedFilename = M + self.filename
        writeToFile(self.modifiedFilename, self.replacedPaddedLines)


class Unigram:
    """ counting padding and unk as a token and map them to a dictionary keys and have value as their
        probability under this model"""

    def __init__(self, pre, smoothing=False):
        self.ungTokenMap = pre.replacedTokensMap
        self.ungTotalToken = pre.replacedTotalTokens
        self.ungUniqueToken = pre.replacedUniqueTokens
        self.ungProbabilityMap = dict()
        if smoothing:
            for token in self.ungUniqueToken.keys():
                self.ungProbabilityMap[token] = (self.ungTokenMap.get(token, 0)+1)/(self.ungUniqueToken+self.ungTotalToken)
        else:
            for token in self.ungTokenMap.keys():
                self.ungProbabilityMap[token] = self.calUngWordProb(token)

    def calUngWordProb(self, word):
        """:returns probability of the given word"""

        return self.ungTokenMap.get(word, 0)/self.ungTotalToken

    def calUniSentProb(self, sentence):
        """:returns the probability of a given sentence"""

        prob = 1
        words = (START + " " + sentence + " " + END).lower().split()
        for word in words:
            prob *= self.ungProbabilityMap[word]
        return prob


class Bigram(Unigram):
    """counting padding and unk as a token and map them to a dictionary keys and have value as
    their probability under this model"""

    def __init__(self, pre):
        Unigram.__init__(self, pre)
        self.biTokenMap = makeBigramMap(pre.replacedPaddedLines)

    def calBiWordProb(self, previousWord, word):
        top = self.biTokenMap.get((previousWord, word), 0)
        bottom = self.ungTokenMap[previousWord]
        return top/bottom

    def calBiSentProb(self, sentence):
        prob = 0
        words = (START + " " + sentence + " " + END).lower().split()
        j = 1
        for i in range(len(words)-1):
            w1 = words[i]
            w2 = words[j]
            j += 1
            prob += self.calBiWordProb(w1, w2)
        return prob


class BigramSmoothing(Unigram):
    def __init__(self, pre):
        Unigram.__init__(self, pre, True)
        self.bisTokensMap = makeBigramMap(pre.replacedPaddedLines)

    def calBisWordProb(self, previousWord, word):
        top = self.bisTokensMap.get((previousWord, word), 0) + 1
        bottom = self.ungTokenMap[previousWord] + len(self.bisTokensMap.keys())
        return top/bottom

    def calBisSentProb(self, sentence):
        prob = 0
        words = (START + " " + sentence + " " + END).lower().split()
        j = 1
        for i in range(len(words)-1):
            w1 = words[i]
            w2 = words[j]
            j += 1
            prob += self.calBisWordProb(w1, w2)
        return prob

