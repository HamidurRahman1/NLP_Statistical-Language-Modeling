
from QA import qa1
from QA import qa2
from QA import qa3
from QA import qa4
from QA import qa5

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

qa1(brownTraining)
qa2(brownTraining)
qa3(brownTraining, brownTest, learnerTest)
qa4(bigramBrownTraining, brownTraining, brownTest, learnerTest)



