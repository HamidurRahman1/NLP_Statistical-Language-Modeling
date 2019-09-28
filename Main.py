
from QA import qa1
from QA import qa2
from QA import qa3
from QA import qa4

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

uni = Unigram(btr)
bi = Bigram(btr)
bis = BigramSmoothing(btr)

print(btr.filename)







# qa1(btr)
# qa2(btr)
# qa3(btr, bts, lts)
# qa4(bi)


