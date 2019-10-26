
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

from Util import S1
from Util import S2
from Util import S3

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
qa5([S1, S2, S3], unigramBrownTraining, bigramBrownTraining, bigramSmoothingBrownTraining)
qa6([S1, S2, S3], unigramBrownTraining, bigramBrownTraining, bigramSmoothingBrownTraining)
qa7(unigramBrownTraining, bigramBrownTraining, bigramSmoothingBrownTraining, brownTest, learnerTest)

