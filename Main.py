
from QA import qa1
from QA import qa2
from QA import qa3
from QA import qa4
from QA import qa5

from Classes import PreProcess
from Classes import Unigram
from Classes import Bigram
from Classes import BigramSmoothing

from Util import RF_B_Tr
from Util import RF_B_Ts
from Util import RF_L_Ts

btr = PreProcess(RF_B_Tr)
bts = PreProcess(RF_B_Ts, btr)
lts = PreProcess(RF_L_Ts, btr)

uni_btr = Unigram(btr)
bi_btr = Bigram(btr)
bis_btr = BigramSmoothing(btr)

qa5(uni_btr, bi_btr, bis_btr)
