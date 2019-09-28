
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

# uni_btr = Unigram(btr)
# bi_btr = Bigram(btr)
# bis_btr = BigramSmoothing(btr)
#
# uni_bts = Unigram(btr)
# bi_bts = Bigram(btr)
# bis_bts = BigramSmoothing(btr)
#
# uni_lts = Unigram(btr)
# bi_lts = Bigram(btr)
# bis_lts = BigramSmoothing(btr)
#
#

print(btr.filename)







# qa1(btr)
# qa2(btr)
# qa3(btr, bts, lts)
# qa4(bi)


