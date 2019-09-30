
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


brownTraining = PreProcess(BROWN_TRAINING)
brownTest = PreProcess(BROWN_TEST, brownTraining)
learnerTest = PreProcess(LEARNER_TEST, brownTraining)

unigramBrownTraining = Unigram(brownTraining)
bigramBrownTraining = Bigram(brownTraining)
bigramSmoothingBrownTraining = BigramSmoothing(brownTraining)

s1 = "He was laughed off the screen ."
s2 = "There was no compulsion behind them ."
s3 = "I look forward to hearing your reply ."

qa7(unigramBrownTraining, bigramBrownTraining, bigramSmoothingBrownTraining, brownTest, learnerTest)

# i = 0
# probTotal = 0
# for line in brownTest.replacedLines:
#     prob = unigramBrownTraining.calUniSentPerplexityTest(line, True)
#     probTotal += prob
# print(probTotal)
# print(math.pow(2, -(probTotal/brownTest.replacedTotalToken)))

