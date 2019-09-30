
from Util import getNonMatching
from Util import makeBigramMap
from Util import getPercentage
from Util import testFileProbabilityOfSentences
from Util import returnLogProbability

import math

t = "\t\t"


def qa1(brownTraining):
    print("Ans #1: unique tokens in training file with <s>, </s> and <unk> :", brownTraining.replacedUniqueToken)


def qa2(brownTraining):
    print("Ans #2: total tokens in training file with :", brownTraining.actualTotalToken)


def qa3(brownTraining, brownTest, learnerTest):
    print("Ans #3: non matching percentage of types and tokens of tests with training before mapping <unk> :")

    t1 = getNonMatching(brownTraining.actualTokenMap.keys(), brownTest.actualTokenMap.keys(), brownTest.actualTokenMap)
    print(t, brownTraining.filename, "->", brownTest.filename, ":", t1, "types % ", getPercentage(t1[0], brownTest.actualUniqueToken),
          "-> tokens % ", getPercentage(t1[1], brownTest.actualTotalToken))

    t2 = getNonMatching(brownTraining.actualTokenMap.keys(), learnerTest.actualTokenMap.keys(), learnerTest.actualTokenMap)
    print(t, brownTraining.filename, "->", learnerTest.filename, ":", t2, "types % ", getPercentage(t2[0], learnerTest.actualUniqueToken),
          "-> tokens % ", getPercentage(t2[1], learnerTest.actualTotalToken))


def qa4(bigramBrownTraining, brownTraining, brownTest, learnerTest):
    print("Ans #4: non matching percentage Bigram of types and tokens of tests with training including <unk> :")

    bigramBrownTest = makeBigramMap(brownTest.replacedLines)
    bigramLearnerTest = makeBigramMap(learnerTest.replacedLines)

    t1 = getNonMatching(bigramBrownTraining.biTokenMap.keys(), bigramBrownTest.keys(), bigramBrownTest)
    print(t, brownTraining.modifiedFilename, "->", brownTest.modifiedFilename, ":", t1, "types % ", getPercentage(t1[0], len(bigramBrownTest)),
          "-> tokens % ", getPercentage(t1[1], sum(bigramBrownTest.values())))

    t2 = getNonMatching(bigramBrownTraining.biTokenMap.keys(), bigramLearnerTest.keys(), bigramLearnerTest)
    print(t, brownTraining.modifiedFilename, "->", learnerTest.modifiedFilename, ":", t2, "types % ", getPercentage(t2[0], len(bigramLearnerTest)),
          "-> tokens % ", getPercentage(t2[1], sum(bigramLearnerTest.values())))


def qa5(unigramBrownTraining, bigramBrownTraining, bigramSmoothingBrownTraining):
    s1 = "He was laughed off the screen ."
    s2 = "There was no compulsion behind them ."
    s3 = "I look forward to hearing your reply ."
    print("Ans #5: Log probabilities of below sentences under 3 models.")

    print(t, s1, "-> Under Unigram: ", returnLogProbability(unigramBrownTraining.calUniSentProb(s1)))
    print(t, s1, "-> Under Bigram: ", returnLogProbability(bigramBrownTraining.calBiSentProb(s1)))
    print(t, s1, "-> Under Bigram Smoothing: ", returnLogProbability(bigramSmoothingBrownTraining.calBisSentProb(s1)))
    print()
    print(t, s2, "-> Under Unigram: ", returnLogProbability(unigramBrownTraining.calUniSentProb(s2)))
    print(t, s2, "-> Under Bigram: ", returnLogProbability(bigramBrownTraining.calBiSentProb(s2)))
    print(t, s2, "-> Under Bigram Smoothing: ", returnLogProbability(bigramSmoothingBrownTraining.calBisSentProb(s2)))
    print()
    print(t, s3, "-> Under Unigram: ", returnLogProbability(unigramBrownTraining.calUniSentProb(s3)))
    print(t, s3, "-> Under Bigram: ", returnLogProbability(bigramBrownTraining.calBiSentProb(s3)))
    print(t, s3, "-> Under Bigram Smoothing: ", returnLogProbability(bigramSmoothingBrownTraining.calBisSentProb(s3)))


def qa6():
    pass


def qa7(brownTest, learnerTest, models):

    m = len(brownTest.replacedLines)
    allSentProbability = testFileProbabilityOfSentences(brownTest.replacedLines, brownTest, models)
    print(allSentProbability)
    log = math.log(allSentProbability, 2)
    l = (1/m)*(log)
    print(math.pow(2, -l))

    # 0. m, total sentence in the test
    # 1. get all sentence probability
    # 2. get the log base 2 of 1
    # 3. calculate l = 1/m * step 2
    # 4. 2^-l

    # pass

