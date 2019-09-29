
from Util import getNonMatching
from Util import makeBigramMap
from Util import getPercentage

import math

t = "\t\t"


def qa1(btr):
    print("Ans #1: unique tokens in training file with <s>, </s> and <unk> :", btr.replacedUniqueToken)


def qa2(btr):
    print("Ans #2: total tokens in training file with :", btr.actualTotalToken)


def qa3(btr, bts, lts):
    print("Ans #3: non matching percentage of types and tokens of tests with training before mapping <unk> :")

    t1 = getNonMatching(btr.actualTokenMap.keys(), bts.actualTokenMap.keys(), bts.actualTokenMap)
    print(t, btr.filename, "->", bts.filename, ":", t1, "types % ", getPercentage(t1[0], bts.actualUniqueToken),
          "-> tokens % ", getPercentage(t1[1], bts.actualTotalToken))

    t2 = getNonMatching(btr.actualTokenMap.keys(), lts.actualTokenMap.keys(), lts.actualTokenMap)
    print(t, btr.filename, "->", bts.filename, ":", t2, "types % ", getPercentage(t2[0], lts.actualUniqueToken),
          "-> tokens % ", getPercentage(t2[1], lts.actualTotalToken))


def qa4(bi_btr, btr, bts, lts):
    print("Ans #4: non matching percentage Bigram of types and tokens of tests with training including <unk> :")

    bi_bts = makeBigramMap(bts.replacedLines)
    bi_lts = makeBigramMap(lts.replacedLines)

    t1 = getNonMatching(bi_btr.biTokenMap.keys(), bi_bts.keys(), bi_bts)
    print(t, btr.modifiedFilename, "->", bts.modifiedFilename, ":", t1, "types % ", getPercentage(t1[0], len(bi_bts)),
          "-> tokens % ", getPercentage(t1[1], sum(bi_bts.values())))

    t2 = getNonMatching(bi_btr.biTokenMap.keys(), bi_lts.keys(), bi_lts)
    print(t, btr.modifiedFilename, "->", lts.modifiedFilename, ":", t2, "types % ", getPercentage(t2[0], len(bi_lts)),
          "-> tokens % ", getPercentage(t2[1], sum(bi_lts.values())))


def qa5(uni, bi, bis):
    s1 = "He was laughed off the screen ."
    s2 = "There was no compulsion behind them ."
    s3 = "I look forward to hearing your reply ."
    print("Ans #5: Log probabilities of below sentences under 3 models.")

    print(t, s1, "-> Under Unigram: ", uni.calUniSentProb(s1))
    print(t, s1, "-> Under Bigram: ", bi.calBiSentProb(s1))
    print(t, s1, "-> Under Bigram Smoothing: ", bis.calBisSentProb(s1))
    print()
    print(t, s2, "-> Under Unigram: ", uni.calUniSentProb(s2))
    print(t, s2, "-> Under Bigram: ", bi.calBiSentProb(s2))
    print(t, s2, "-> Under Bigram Smoothing: ", bis.calBisSentProb(s2))
    print()
    print(t, s3, "-> Under Unigram: ", uni.calUniSentProb(s3))
    print(t, s3, "-> Under Bigram: ", bi.calBiSentProb(s3))
    print(t, s3, "-> Under Bigram Smoothing: ", bis.calBisSentProb(s3))


def qa6():
    pass


def qa7():
    pass

