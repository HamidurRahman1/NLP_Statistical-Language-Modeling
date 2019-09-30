
import math

from QA import qa1
from QA import qa2
from QA import qa3
from QA import qa4
from QA import qa5
from QA import qa6
from QA import qa7

from Classes import PreProcess
from Classes import Unigram
from Classes import Bigram
from Classes import BigramSmoothing

from Util import BROWN_TRAINING
from Util import BROWN_TEST
from Util import LEARNER_TEST

from Util import makeWords
from Util import countWords

brownTraining = PreProcess(BROWN_TRAINING)
brownTest = PreProcess(BROWN_TEST, brownTraining)
learnerTest = PreProcess(LEARNER_TEST, brownTraining)

unigramBrownTraining = Unigram(brownTraining)
bigramBrownTraining = Bigram(brownTraining)
bigramSmoothingBrownTraining = BigramSmoothing(brownTraining)

qa5(unigramBrownTraining, bigramBrownTraining, bigramSmoothingBrownTraining)

# print(brownTest.)
# print(brownTest.replacedUniqueToken)
#
# print(brownTest.replacedUniqueToken)
# print(learnerTest.replacedUniqueToken)


# s1 = "He was laughed off the screen ."
# s2 = "There was no compulsion behind them ."
# s3 = "I look forward to hearing your reply ."
#
# p1 = unigramBrownTraining.calUniSentProb(s1)
# p2 = unigramBrownTraining.calUniSentProb(s2)
# p3 = unigramBrownTraining.calUniSentProb(s3)
# print(math.log(p1, 2))
# print(p2)
# print(math.log(p3, 2))

#qa5(unigramBrownTraining, bigramBrownTraining, bigramSmoothingBrownTraining)

# unigramBrownTest = Unigram(brownTest)
# bigramBrownTest = Bigram(brownTest)
# bigramSmoothingBrownTest = BigramSmoothing(brownTest)
# print("trained Brown Test")
#
# unigramLearnerTest = Unigram(learnerTest)
# bigramLearnerTest = Bigram(learnerTest)
# bigramSmoothingLearnerTest = BigramSmoothing(learnerTest)
# print("trained Learner Test")


