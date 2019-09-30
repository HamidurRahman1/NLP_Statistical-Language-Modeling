
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


# unigramBrownTest = Unigram(brownTest)
# bigramBrownTest = Bigram(brownTest)
# bigramSmoothingBrownTest = BigramSmoothing(brownTest)
# print("trained Brown Test")
#
# unigramLearnerTest = Unigram(learnerTest)
# bigramLearnerTest = Bigram(learnerTest)
# bigramSmoothingLearnerTest = BigramSmoothing(learnerTest)
# print("trained Learner Test")


