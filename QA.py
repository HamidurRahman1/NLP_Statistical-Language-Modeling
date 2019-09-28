
from Util import getNonMatching

t = "\t\t"


def qa1(btr):
    print("Ans #1: unique tokens in training file with <s>, </s> and <unk> :", btr.actualUniqueToken)


def qa2(btr):
    print("Ans #2: total tokens in training file with :", btr.actualTotalToken)


def qa3(btr, bts, lts):
    print("Ans #3: non matching percentage of types and tokens of tests with training :")
    print(t, btr.filename, "->", bts.filename, ":",
          getNonMatching(btr.initialTokensMap.keys(), bts.initialTokensMap.keys(), bts.initialTokensMap))
    print(t, btr.filename, "->", lts.filename, ":",
          getNonMatching(btr.initialTokensMap.keys(), lts.initialTokensMap.keys(), lts.initialTokensMap))


def qa4(bi):
    print("Ans #4: non matching percentage of types and tokens of tests with training :")
    for t in bi.biTokenMap:
        t1 = t[0]
        t2 = t[1]
        if t1 == t2:
            print(t, bi.biTokenMap.get(t))


